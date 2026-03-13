#include <boost/ut.hpp>

#include <gnuradio-4.0/NMEAParser.hpp>

using namespace boost::ut;
using namespace gr::timing;

namespace {

std::string nmea(std::string_view body) {
    auto cs = gr::timing::detail::computeNMEAChecksum(body.substr(1)); // skip '$'
    return std::format("{}*{:02X}", body, cs);
}

} // namespace

const boost::ut::suite<"NMEAParser"> nmeaParserTests = [] {
    "parse RMC with valid fix"_test = [] {
        NMEAParser parser;
        auto       fix1 = parser.parseLine(nmea("$GPRMC,120000.00,A,5001.1900,N,00840.6570,E,0.5,54.7,110326,,,A"), 1000);
        expect(!fix1.has_value()) << "first line sets baseline, no PPS yet";

        auto fix2 = parser.parseLine(nmea("$GPRMC,120001.00,A,5001.1910,N,00840.6580,E,1.0,90.0,110326,,,A"), 2000);
        expect(fix2.has_value()) << "second change triggers PPS";

        expect(fix2->hasPosition);
        expect(fix2->hasTime);
        expect(fix2->latitude > 50.f && fix2->latitude < 51.f) << "latitude ~50.02";
        expect(fix2->longitude > 8.f && fix2->longitude < 9.f) << "longitude ~8.68";
        expect(fix2->speedKmh > 0.f) << "speed from knots conversion";
        expect(fix2->utcTimestampNs > 0ULL) << "UTC timestamp set";
        expect(eq(fix2->localTimeNs, 2000ULL)) << "local time preserved";
    };

    "parse RMC with no fix marks unlocked"_test = [] {
        NMEAParser parser;
        expect(!parser.parseLine(nmea("$GPRMC,120000.00,V,,,,,,,110326,,,N"), 1000).has_value()) << "first line sets baseline";
        auto fix = parser.parseLine(nmea("$GPRMC,120001.00,V,,,,,,,110326,,,N"), 2000);
        expect(fix.has_value()) << "PPS still emitted without fix";
        expect(!fix->hasPosition) << "no position when status=V";
        expect(fix->hasTime) << "time is still parsed";
    };

    "parse GGA accumulates position and altitude"_test = [] {
        NMEAParser parser;
        expect(!parser.parseLine(nmea("$GPRMC,120000.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A"), 1000).has_value()) << "RMC sets baseline";
        expect(!parser.parseLine(nmea("$GPGGA,120000.00,5001.1900,N,00840.6570,E,1,10,0.8,136.0,M,47.0,M,,"), 1100).has_value()) << "GGA same second, no PPS";

        auto fix = parser.parseLine(nmea("$GPRMC,120001.00,A,5001.1910,N,00840.6580,E,0.0,0.0,110326,,,A"), 2000);
        expect(fix.has_value());
        expect(eq(fix->satellites, 10));
        expect(fix->hdop > 0.7f && fix->hdop < 0.9f) << "HDOP from GGA";
        expect(fix->altitude > 135.f && fix->altitude < 137.f) << "altitude from GGA";
    };

    "parse GSA sets fix type"_test = [] {
        NMEAParser parser;
        expect(!parser.parseLine(nmea("$GPRMC,120000.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A"), 1000).has_value()) << "RMC sets baseline";
        expect(!parser.parseLine(nmea("$GPGSA,A,3,04,05,09,12,,,,,,,,,1.8,1.0,1.5"), 1100).has_value()) << "GSA same second, no PPS";

        auto fix = parser.parseLine(nmea("$GPRMC,120001.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A"), 2000);
        expect(fix.has_value());
        expect(fix->fixType == FixType::fix3D) << "3D fix from GSA mode=3";
    };

    "parse VTG updates speed and heading"_test = [] {
        NMEAParser parser;
        expect(!parser.parseLine(nmea("$GPRMC,120000.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A"), 1000).has_value()) << "RMC sets baseline";
        expect(!parser.parseLine(nmea("$GPVTG,54.7,T,34.4,M,5.5,N,10.2,K"), 1100).has_value()) << "VTG same second, no PPS";

        auto fix = parser.parseLine(nmea("$GPRMC,120001.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A"), 2000);
        expect(fix.has_value());
        expect(fix->speedKmh > 10.f && fix->speedKmh < 10.5f) << "speed from VTG";
        expect(fix->headingDeg > 54.f && fix->headingDeg < 55.f) << "heading from VTG";
    };

    "no PPS on same second"_test = [] {
        NMEAParser parser;
        expect(!parser.parseLine(nmea("$GPRMC,120000.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A"), 1000).has_value()) << "first RMC sets baseline";
        expect(!parser.parseLine(nmea("$GPRMC,120000.50,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A"), 1500).has_value()) << "same second = no PPS";
    };

    "GGA-only PPS detection"_test = [] {
        NMEAParser parser;
        expect(!parser.parseLine(nmea("$GPGGA,120000.00,5001.1900,N,00840.6570,E,1,8,1.0,100.0,M,47.0,M,,"), 1000).has_value()) << "first GGA sets baseline";
        auto fix = parser.parseLine(nmea("$GPGGA,120001.00,5001.1900,N,00840.6570,E,1,8,1.0,100.0,M,47.0,M,,"), 2000);
        expect(fix.has_value()) << "PPS from GGA second change";
    };

    "empty and malformed lines are ignored"_test = [] {
        NMEAParser parser;
        expect(!parser.parseLine("", 0).has_value()) << "empty line";
        expect(!parser.parseLine("not nmea", 0).has_value()) << "non-NMEA text";
        expect(!parser.parseLine("$GP", 0).has_value()) << "truncated sentence";
        expect(!parser.parseLine("$GPRMC,short", 0).has_value()) << "too few fields";
    };

    "lines without checksum are accepted"_test = [] {
        NMEAParser parser;
        expect(!parser.parseLine("$GPRMC,120000.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A", 1000).has_value()) << "baseline without checksum";
        auto fix = parser.parseLine("$GPRMC,120001.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A", 2000);
        expect(fix.has_value()) << "no checksum accepted";
    };

    "invalid checksum rejects line"_test = [] {
        NMEAParser parser;
        expect(!parser.parseLine(nmea("$GPRMC,120000.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A"), 1000).has_value()) << "baseline";
        auto fix = parser.parseLine("$GPRMC,120001.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A*FF", 2000);
        expect(!fix.has_value()) << "bad checksum rejected";
    };

    "GNSS talker prefixes (GN, GL, GA) are handled"_test = [] {
        NMEAParser parser;
        expect(!parser.parseLine(nmea("$GNRMC,120000.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A"), 1000).has_value()) << "GN baseline";
        auto fix = parser.parseLine(nmea("$GNRMC,120001.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A"), 2000);
        expect(fix.has_value()) << "GN talker prefix works";
    };

    "coordinate parsing covers hemispheres"_test = [] {
        NMEAParser parser;
        expect(!parser.parseLine(nmea("$GPRMC,120000.00,A,3345.1234,S,15110.5678,W,0.0,0.0,110326,,,A"), 1000).has_value()) << "baseline";
        auto fix = parser.parseLine(nmea("$GPRMC,120001.00,A,3345.1234,S,15110.5678,W,0.0,0.0,110326,,,A"), 2000);
        expect(fix.has_value());
        expect(fix->latitude < 0.f) << "south is negative";
        expect(fix->longitude < 0.f) << "west is negative";
    };

    "nmeaToUTCNs produces correct timestamp"_test = [] {
        auto ns = gr::timing::detail::nmeaToUTCNs("120000.00", "110326");
        expect(ns > 0ULL);

        using namespace std::chrono;
        auto tp       = sys_days{2026y / March / 11} + 12h;
        auto expected = static_cast<std::uint64_t>(duration_cast<nanoseconds>(tp.time_since_epoch()).count());
        expect(eq(ns, expected)) << "timestamp matches 2026-03-11T12:00:00Z";
    };

    "parseField handles invalid input"_test = [] {
        expect(eq(gr::timing::detail::parseField<int>(""), 0));
        expect(eq(gr::timing::detail::parseField<float>("abc"), 0.f));
        expect(eq(gr::timing::detail::parseField<int>("42"), 42));
    };

    "computeNMEAChecksum XORs payload bytes"_test = [] {
        // "GP" → 'G' ^ 'P' = 0x47 ^ 0x50 = 0x17
        expect(eq(gr::timing::detail::computeNMEAChecksum("GP"), std::uint8_t{0x17}));
        expect(eq(gr::timing::detail::computeNMEAChecksum(""), std::uint8_t{0}));
    };

    "validateNMEAChecksum accepts valid and rejects invalid"_test = [] {
        auto valid = nmea("$GPRMC,120000.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A");
        expect(gr::timing::detail::validateNMEAChecksum(valid)) << "valid checksum accepted";
        expect(!gr::timing::detail::validateNMEAChecksum("$GPRMC,test*FF")) << "wrong checksum rejected";
        expect(gr::timing::detail::validateNMEAChecksum("$GPRMC,test")) << "no checksum accepted";
        expect(!gr::timing::detail::validateNMEAChecksum("$GP*")) << "incomplete checksum rejected";
    };

    "lastFix returns most recent completed fix"_test = [] {
        NMEAParser parser;
        expect(!parser.parseLine(nmea("$GPRMC,120000.00,A,5001.1900,N,00840.6570,E,0.0,0.0,110326,,,A"), 1000).has_value()) << "baseline";
        expect(parser.lastFix().utcTimestampNs == 0ULL) << "no complete fix yet";

        auto fix = parser.parseLine(nmea("$GPRMC,120001.00,A,5001.1910,N,00840.6580,E,0.0,0.0,110326,,,A"), 2000);
        expect(fix.has_value()) << "PPS triggered";
        expect(parser.lastFix().utcTimestampNs > 0ULL) << "fix available after PPS";
        expect(parser.lastFix().hasPosition);
    };
};

int main() { return 0; }
