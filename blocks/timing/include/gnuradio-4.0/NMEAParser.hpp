#ifndef GNURADIO_NMEA_PARSER_HPP
#define GNURADIO_NMEA_PARSER_HPP

#include <charconv>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gr::timing {

// NMEA talker IDs: GP=GPS, GL=GLONASS, GA=Galileo, GN=mixed GNSS, BD=BeiDou
// see https://gpsd.gitlab.io/gpsd/NMEA.html

enum class FixType : std::uint8_t { none = 0, fix2D = 2, fix3D = 3 };
enum class EmitMode : std::uint8_t { ppsOnly, clock };

struct GpsFix {
    std::uint64_t utcTimestampNs = 0; // GPS-derived UTC [ns since epoch]
    std::uint64_t localTimeNs    = 0; // host wall-clock when serial data arrived [ns since epoch]
    float         latitude       = 0.f;
    float         longitude      = 0.f;
    float         altitude       = 0.f; // metres above MSL
    std::int32_t  satellites     = 0;
    float         hdop           = 0.f;
    FixType       fixType        = FixType::none;
    float         speedKmh       = 0.f;
    float         headingDeg     = 0.f;
    bool          hasPosition    = false;
    bool          hasTime        = false;
};

namespace detail {

inline std::vector<std::string_view> splitNMEAFields(std::string_view line) {
    std::vector<std::string_view> fields;
    fields.reserve(20);
    std::size_t start = 0;
    while (true) {
        auto end = line.find(',', start);
        fields.push_back(line.substr(start, end - start));
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return fields;
}

template<typename T>
T parseField(std::string_view s, T fallback = {}) {
    T val{};
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
    return ec == std::errc{} ? val : fallback;
}

// NMEA coordinates: DDMM.MMMM (lat) or DDDMM.MMMM (lon) → decimal degrees
inline float parseNMEACoord(std::string_view coord, std::string_view hemi) {
    if (coord.empty()) {
        return 0.f;
    }
    auto dot = coord.find('.');
    if (dot == std::string_view::npos || dot < 2) {
        return 0.f;
    }

    const std::size_t degDigits = (dot > 4) ? 3UZ : 2UZ;
    int               deg       = parseField<int>(coord.substr(0, degDigits));
    float             minutes   = parseField<float>(coord.substr(degDigits));
    float             decimal   = static_cast<float>(deg) + minutes / 60.f;
    if (hemi == "S" || hemi == "W") {
        decimal = -decimal;
    }
    return decimal;
}

// NMEA time "HHMMSS.ss" + date "DDMMYY" → nanoseconds since Unix epoch
inline std::uint64_t nmeaToUTCNs(std::string_view time, std::string_view date) {
    if (time.size() < 6) {
        return 0;
    }

    int h = parseField<int>(time.substr(0, 2));
    int m = parseField<int>(time.substr(2, 2));
    int s = parseField<int>(time.substr(4, 2));

    double fractionalSeconds = 0.0;
    if (time.size() > 6 && time[6] == '.') {
        fractionalSeconds = parseField<double>(time.substr(6));
    }

    int day = 1, month = 1, year = 2000;
    if (date.size() == 6) {
        day   = parseField<int>(date.substr(0, 2));
        month = parseField<int>(date.substr(2, 2));
        year  = 2000 + parseField<int>(date.substr(4, 2));
    }

    // days since epoch (simplified — no leap second handling, UTC only)
    std::chrono::year_month_day ymd{std::chrono::year{year}, std::chrono::month{static_cast<unsigned>(month)}, std::chrono::day{static_cast<unsigned>(day)}};
    auto                        daysSinceEpoch = std::chrono::sys_days{ymd}.time_since_epoch();
    auto                        totalNs        = std::chrono::duration_cast<std::chrono::nanoseconds>(daysSinceEpoch);
    totalNs += std::chrono::hours{h} + std::chrono::minutes{m} + std::chrono::seconds{s};
    totalNs += std::chrono::nanoseconds{static_cast<std::int64_t>(fractionalSeconds * 1e9)};
    return static_cast<std::uint64_t>(totalNs.count());
}

inline std::uint64_t wallClockNs() { return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count()); }

inline std::uint8_t computeNMEAChecksum(std::string_view payload) {
    std::uint8_t cs = 0;
    for (char c : payload) {
        cs ^= static_cast<std::uint8_t>(c);
    }
    return cs;
}

inline bool validateNMEAChecksum(std::string_view line) {
    if (line.size() < 4 || line[0] != '$') {
        return false;
    }
    auto star = line.rfind('*');
    if (star == std::string_view::npos) {
        return true; // no checksum field — accept (some devices omit it)
    }
    if (star + 3 != line.size()) {
        return false; // checksum must be exactly 2 hex digits at end of line
    }
    std::uint8_t computed = computeNMEAChecksum(line.substr(1, star - 1));
    std::uint8_t expected = 0;
    auto [ptr, ec]        = std::from_chars(line.data() + star + 1, line.data() + star + 3, expected, 16);
    return ec == std::errc{} && computed == expected;
}

} // namespace detail

/// Incremental NMEA 0183 sentence parser. Feed serial lines via parseLine(); returns a
/// completed GpsFix each time a PPS boundary (UTC second change) is detected.
struct NMEAParser {
    GpsFix      _pendingFix;
    GpsFix      _lastCompleteFix;
    int         _lastUtcSecond = -1;
    std::string _lastDate; // cached date from RMC for GGA time resolution

    // returns the completed fix when a PPS boundary is detected (UTC second changed)
    [[nodiscard]] std::optional<GpsFix> parseLine(std::string_view line, std::uint64_t localTimeNs) {
        if (line.empty() || line[0] != '$') {
            return std::nullopt;
        }
        if (line.back() == '\r') {
            line.remove_suffix(1);
        }
        if (!detail::validateNMEAChecksum(line)) {
            return std::nullopt;
        }
        auto star = line.rfind('*');
        if (star != std::string_view::npos) {
            line = line.substr(0, star);
        }

        auto sentenceId = extractSentenceId(line);
        if (sentenceId.empty()) {
            return std::nullopt;
        }

        auto fields = detail::splitNMEAFields(line);

        if (sentenceId == "RMC") {
            return parseRMC(fields, localTimeNs);
        } else if (sentenceId == "GGA") {
            return parseGGA(fields, localTimeNs);
        } else if (sentenceId == "VTG") {
            parseVTG(fields);
            return std::nullopt;
        } else if (sentenceId == "GSA") {
            parseGSA(fields);
            return std::nullopt;
        }
        return std::nullopt;
    }

    [[nodiscard]] const GpsFix& lastFix() const noexcept { return _lastCompleteFix; }

private:
    // extract "RMC" from "$GPRMC,..." or "$GNRMC,..."
    static std::string_view extractSentenceId(std::string_view line) {
        if (line.size() < 6 || line[0] != '$') {
            return {};
        }
        auto comma = line.find(',');
        if (comma == std::string_view::npos || comma < 4) {
            return {};
        }
        return line.substr(3, comma - 3); // skip "$XX" talker prefix
    }

    std::optional<GpsFix> checkPpsBoundary(std::string_view timeField, std::uint64_t localTimeNs) {
        if (timeField.size() < 6) {
            return std::nullopt;
        }
        int utcSecond = detail::parseField<int>(timeField.substr(4, 2));

        std::optional<GpsFix> result;
        if (_lastUtcSecond >= 0 && utcSecond != _lastUtcSecond) {
            _lastCompleteFix             = _pendingFix;
            _lastCompleteFix.localTimeNs = localTimeNs;
            result                       = _lastCompleteFix;
            _pendingFix                  = GpsFix{};
        }
        _lastUtcSecond = utcSecond;
        return result;
    }

    // $xxRMC — recommended minimum specific GNSS data
    // $GPRMC,225444,A,4916.45,N,12311.12,W,000.5,054.7,191194,020.3,E*68
    //         |     | |       |    |       |   |     |     |      |
    //         |     | |       |    |       |   |     |     |      └── magnetic variation
    //         |     | |       |    |       |   |     |     └──────── date (ddmmyy)
    //         |     | |       |    |       |   |     └────────────── course over ground
    //         |     | |       |    |       |   └──────────────────── speed over ground (knots)
    //         |     | |       |    └──────────────────────────────── position
    //         |     | └───────────────────────────────────────────── fix status (A=valid)
    //         └───────────────────────────────────────────────────── UTC time
    std::optional<GpsFix> parseRMC(const std::vector<std::string_view>& f, std::uint64_t localTimeNs) {
        if (f.size() < 12) {
            return std::nullopt;
        }

        auto ppsFix = checkPpsBoundary(f[1], localTimeNs);

        if (f[9].size() == 6) {
            _lastDate = std::string(f[9]);
        }

        if (f[2] == "A") { // valid fix
            _pendingFix.utcTimestampNs = detail::nmeaToUTCNs(f[1], f[9]);
            _pendingFix.latitude       = detail::parseNMEACoord(f[3], f[4]);
            _pendingFix.longitude      = detail::parseNMEACoord(f[5], f[6]);
            _pendingFix.speedKmh       = detail::parseField<float>(f[7]) * 1.852f; // knots → km/h
            _pendingFix.headingDeg     = detail::parseField<float>(f[8]);
            _pendingFix.hasPosition    = true;
            _pendingFix.hasTime        = true;
        } else {
            _pendingFix.utcTimestampNs = detail::nmeaToUTCNs(f[1], _lastDate);
            _pendingFix.hasTime        = !f[1].empty();
        }
        return ppsFix;
    }

    // $xxGGA — global positioning system fix data
    // $GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47
    //         |      |        |     |        | | |  |   |     |   |
    //         |      |        |     |        | | |  |   |     |   └── geoid separation unit
    //         |      |        |     |        | | |  |   |     └───── geoid separation
    //         |      |        |     |        | | |  |   └──────────── altitude above MSL
    //         |      |        |     |        | | |  └──────────────── HDOP
    //         |      |        |     |        | | └─────────────────── satellites used
    //         |      |        |     |        | └───────────────────── fix quality (0=no fix, 1=GPS)
    //         |      |        |     |        └─────────────────────── longitude + hemisphere
    //         |      |        └────────────────────────────────────── latitude + hemisphere
    //         └────────────────────────────────────────────────────── UTC time
    std::optional<GpsFix> parseGGA(const std::vector<std::string_view>& f, std::uint64_t localTimeNs) {
        if (f.size() < 15) {
            return std::nullopt;
        }

        auto ppsFix = checkPpsBoundary(f[1], localTimeNs);

        int fixQuality = detail::parseField<int>(f[6]);
        if (fixQuality > 0) {
            _pendingFix.latitude    = detail::parseNMEACoord(f[2], f[3]);
            _pendingFix.longitude   = detail::parseNMEACoord(f[4], f[5]);
            _pendingFix.satellites  = detail::parseField<std::int32_t>(f[7]);
            _pendingFix.hdop        = detail::parseField<float>(f[8]);
            _pendingFix.altitude    = detail::parseField<float>(f[9]);
            _pendingFix.hasPosition = true;
        }
        if (!_pendingFix.hasTime && !f[1].empty()) {
            _pendingFix.utcTimestampNs = detail::nmeaToUTCNs(f[1], _lastDate);
            _pendingFix.hasTime        = true;
        }
        return ppsFix;
    }

    // $xxVTG — track made good and ground speed
    // $GPVTG,054.7,T,034.4,M,005.5,N,010.2,K*48
    //         |     |  |     |  |     |  |     |
    //         |     |  |     |  |     |  |     └── speed over ground (km/h)
    //         |     |  |     |  |     |  └──────── speed over ground (knots)
    //         |     |  |     |  └───────────────── magnetic track
    //         |     |  └────────────────────────── magnetic indicator
    //         └─────────────────────────────────── true track
    void parseVTG(const std::vector<std::string_view>& f) {
        if (f.size() < 9) {
            return;
        }
        _pendingFix.headingDeg = detail::parseField<float>(f[1]);
        _pendingFix.speedKmh   = detail::parseField<float>(f[7]);
    }

    // $xxGSA — GPS DOP and active satellites
    // $GPGSA,A,3,04,05,...,29,1.8,1.0,1.5
    //         | | |                |   |   |
    //         | | |                |   |   └── VDOP
    //         | | |                |   └────── HDOP
    //         | | |                └────────── PDOP
    //         | | └──────────────────────────── fix type (1=none, 2=2D, 3=3D)
    //         | └────────────────────────────── mode (A=auto, M=manual)
    //         └──────────────────────────────── sentence ID
    void parseGSA(const std::vector<std::string_view>& f) {
        if (f.size() < 18) {
            return;
        }
        int mode            = detail::parseField<int>(f[2]);
        _pendingFix.fixType = (mode == 3) ? FixType::fix3D : (mode == 2) ? FixType::fix2D : FixType::none;
        _pendingFix.hdop    = detail::parseField<float>(f[16]);
    }
};

} // namespace gr::timing

#endif // GNURADIO_NMEA_PARSER_HPP
