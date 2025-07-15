// g++ -std=c++23 -O2 gps_reader_ext.cpp -o gps_reader_ext
// or:
// em++ gps_reader_ext.cpp -std=c++23 -O2 -sEXPORTED_RUNTIME_METHODS=['ccall'] -sALLOW_MEMORY_GROWTH=1 -o gps_reader_ext.html
// emrun --no_browser --port 8080 .
// python3 -m http.server 8080

#include <charconv>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <optional>
#include <poll.h>
#include <print>
#include <string>
#include <string_view>
#include <system_error>
#include <termios.h>
#include <unistd.h>
#include <vector>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#include <emscripten/html5.h>

#include <cstdlib> // for std::strtod

namespace compat {
struct from_chars_result {
    const char* ptr;
    std::errc   ec;
};

inline from_chars_result from_chars(const char* first, const char* last, double& value) {
    char* end = nullptr;
    value     = std::strtod(first, &end);
    if (end == first) {
        return {first, std::errc::invalid_argument};
    } else if (end > last) {
        return {last, std::errc::result_out_of_range};
    }
    return {end, std::errc{}};
}

inline from_chars_result from_chars(const char* first, const char* last, int& value) {
    char* end = nullptr;
    value     = std::strtol(first, &end, 10);
    if (end == first) {
        return {first, std::errc::invalid_argument};
    } else if (end > last) {
        return {last, std::errc::result_out_of_range};
    }
    return {end, std::errc{}};
}
} // namespace compat

#else
#define EMSCRIPTEN_KEEPALIVE inline
#endif

#if defined(__EMSCRIPTEN__)
using compat::from_chars;
#else
using std::from_chars;
#endif

// Optional: log to GPX
std::ofstream gpx("tracklog.gpx", std::ios::app);

struct FixData {
    std::string timestamp;
    double      latitude = 0.0;
    std::string latHemi;
    double      longitude = 0.0;
    std::string lonHemi;
    double      speedKmh   = 0.0;
    double      headingDeg = 0.0;
    double      altitude   = 0.0;
    int         satellites = 0;
    double      hdop       = 0.0;
    std::string fixType;
};

enum class TalkerId {
    GP, // GPS, USA
    GL, // GLONASS, Russia
    GA, // Galileo, Europe
    GN, // Mixed GNSS (all the above)
    BD  // BeiDou, China
};

inline TalkerId getTalkerId(std::string_view line) {
    if (line.size() < 6 || line[0] != '$') {
        return TalkerId::GN;
    }
    auto prefix = line.substr(1, 2);
    if (prefix == "GP") {
        return TalkerId::GP;
    }
    if (prefix == "GL") {
        return TalkerId::GL;
    }
    if (prefix == "GA") {
        return TalkerId::GA;
    }
    if (prefix == "GN") {
        return TalkerId::GN;
    }
    if (prefix == "BD") {
        return TalkerId::BD;
    }
    return TalkerId::GN;
}

// Parse NMEA-style lat/lon coordinate to decimal degrees
// https://gpsd.gitlab.io/gpsd/NMEA.html
// Format: DDMM.MMMM (lat) or DDDMM.MMMM (lon)
double parseCoord(std::string_view coord, std::string_view hemi) {
    if (coord.empty()) {
        return 0.0;
    }
    auto dot = coord.find('.');
    if (dot == std::string_view::npos || dot < 2) {
        return 0.0;
    }
    const auto  degDigits = (dot > 4) ? 3 : 2; // heuristic: lat = 2, lon = 3
    std::string degStr(coord.substr(0, degDigits));
    std::string minStr(coord.substr(degDigits));
    int         deg     = 0;
    double      minutes = 0.0;
    from_chars(degStr.data(), degStr.data() + degStr.size(), deg);
    from_chars(minStr.data(), minStr.data() + minStr.size(), minutes);
    double decimal = deg + (minutes / 60.0);
    if (hemi == "S" || hemi == "W") {
        decimal *= -1.0;
    }
    return decimal;
}

std::string isoTimestamp(std::string_view time, std::string_view date) {
    if (time.size() < 6 || date.size() != 6) {
        return {};
    }

    auto h = time.substr(0, 2);
    auto m = time.substr(2, 2);
    auto s = time.substr(4, 2);

    std::string ms;
    if (time.size() > 6 && time[6] == '.') {
        ms = time.substr(6); // includes the '.' and digits
    }

    return std::format("20{}-{}-{}T{}:{}:{}{}Z", date.substr(4, 2), date.substr(2, 2), date.substr(0, 2), h, m, s, ms);
}

void writeGpx(const FixData& fix) {
    if (!gpx.is_open()) {
        return;
    }
    gpx << std::format("  <trkpt latitude=\"{}\" longitude=\"{}\"><ele>{}</ele><time>{}</time></trkpt>\n", fix.latitude, fix.longitude, fix.altitude, fix.timestamp);
}

void printFix(const FixData& fix) { //
    std::println("{} | {:>12.8f}°{} {:>12.8f}°{} | alt: {:.1f} m | nSat: {} | hdop: {:.1f} | spd: {:.1f} km/h | hdg: {:.1f}° | fix: {}", //
                 fix.timestamp, fix.latitude, fix.latHemi, fix.longitude, fix.lonHemi, fix.altitude, fix.satellites, fix.hdop, fix.speedKmh, fix.headingDeg, fix.fixType);

}

/**
 * $GPGGA - Global Positioning System Fix Data
 * Format:
 * $GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47
 *         |      |        |     |        | | |  |   |     |   |
 *         |      |        |     |        | | |  |   |     |   └── Geoid separation unit
 *         |      |        |     |        | | |  |   |     └───── Geoid separation
 *         |      |        |     |        | | |  |   └──────────── Altitude above MSL
 *         |      |        |     |        | | |  └──────────────── HDOP
 *         |      |        |     |        | | └─────────────────── Satellites used
 *         |      |        |     |        | └───────────────────── Fix quality (0=no fix, 1=GPS)
 *         |      |        |     |        └─────────────────────── Longitude + hemisphere
 *         |      |        └────────────────────────────────────── Latitude + hemisphere
 *         └────────────────────────────────────────────────────── UTC time
 */
std::optional<FixData> parseGGA(const std::vector<std::string_view>& f, FixData& fix) {
    if (f.size() < 15) {
        return std::nullopt;
    }
    fix.latitude  = parseCoord(f[2], f[3]);
    fix.latHemi   = std::string(f[3]);
    fix.longitude = parseCoord(f[4], f[5]);
    fix.lonHemi   = std::string(f[5]);
    from_chars(f[7].data(), f[7].data() + f[7].size(), fix.satellites);
    from_chars(f[8].data(), f[8].data() + f[8].size(), fix.hdop);
    from_chars(f[9].data(), f[9].data() + f[9].size(), fix.altitude);
    return fix;
}

/**
 * $GPVTG - Track Made Good and Ground Speed
 * Format:
 * $GPVTG,054.7,T,034.4,M,005.5,N,010.2,K*48
 *         |     |  |     |  |     |  |     |
 *         |     |  |     |  |     |  |     └── Speed over ground (km/h)
 *         |     |  |     |  |     |  └──────── Speed over ground (knots)
 *         |     |  |     |  └───────────────── Magnetic track
 *         |     |  └────────────────────────── Magnetic indicator
 *         └─────────────────────────────────── True track
 */
void parseVTG(std::vector<std::string_view> f, FixData& fix) {
    if (f.size() < 9) {
        return;
    }
    from_chars(f[1].data(), f[1].data() + f[1].size(), fix.headingDeg);
    from_chars(f[7].data(), f[7].data() + f[7].size(), fix.speedKmh);
}

/**
 * $GPGSA - GPS DOP and Active Satellites
 * Format:
 * $GPGSA,A,3,04,05,...,29,1.8,1.0,1.5
 *         | | |                |   |   |
 *         | | |                |   |   └── VDOP
 *         | | |                |   └────── HDOP
 *         | | |                └────────── PDOP
 *         | | └──────────────────────────── Fix type (1=none, 2=2D, 3=3D)
 *         | └────────────────────────────── Mode (A=auto, M=manual)
 *         └──────────────────────────────── Sentence ID
 */
void parseGSA(std::vector<std::string_view> f, FixData& fix) {
    if (f.size() < 18) {
        return;
    }
    fix.fixType = (f[2] == "1" ? "None" : (f[2] == "2" ? "2D" : "3D"));
    from_chars(f[16].data(), f[16].data() + f[16].size(), fix.hdop); // prefer GSA HDOP if valid
}

/**
 * $GPGLL - Geographic Position, Latitude/Longitude
 * Format:
 * $GPGLL,4916.45,N,12311.12,W,225444,A*1D
 *         |      |  |        |  |     |  |
 *         |      |  |        |  |     |  └── Checksum
 *         |      |  |        |  |     └──── Fix status (A=valid)
 *         |      |  |        |  └────────── UTC time (hhmmss)
 *         |      |  └────────┴───────────── Longitude + hemisphere
 *         └──────────────────────────────── Latitude + hemisphere
 */
void parseGLL(std::vector<std::string_view> f, FixData& fix) {
    if (f.size() < 7 || f[6] != "A") {
        return;
    }
    fix.latitude  = parseCoord(f[1], f[2]);
    fix.latHemi   = std::string(f[2]);
    fix.longitude = parseCoord(f[3], f[4]);
    fix.lonHemi   = std::string(f[4]);
    fix.timestamp = std::string(f[5]);
}

/**
 * $GPGSV - Satellites in View
 * Format:
 * $GPGSV,3,1,11,07,79,048,45,02,67,308,42,...
 *         | | |  |
 *         | | |  └── Total satellites in view
 *         | | └───── Message number
 *         | └─────── Total messages in sequence
 *         └───────── Talker ID (GPS)
 */
void parseGSV(std::vector<std::string_view> f) {
    if (f.size() < 4) {
        return;
    }
    int totalMsgs = 0, msgNum = 0, numSats = 0;
    from_chars(f[1].data(), f[1].data() + f[1].size(), totalMsgs);
    from_chars(f[2].data(), f[2].data() + f[2].size(), msgNum);
    from_chars(f[3].data(), f[3].data() + f[3].size(), numSats);
    std::println("[GSV] Part {}/{}: {} satellites in view", msgNum, totalMsgs, numSats);
}

/**
 * $GPRMC - Recommended Minimum Specific GNSS Data
 * Format:
 * $GPRMC,225444,A,4916.45,N,12311.12,W,000.5,054.7,191194,020.3,E*68
 *         |     | |       |    |       |   |     |     |      |
 *         |     | |       |    |       |   |     |     |      └── Magnetic variation
 *         |     | |       |    |       |   |     |     └──────── Date (ddmmyy)
 *         |     | |       |    |       |   |     └────────────── Course over ground
 *         |     | |       |    |       |   └──────────────────── Speed over ground (knots)
 *         |     | |       |    └──────────────────────────────── Position
 *         |     | └───────────────────────────────────────────── Fix status (A=valid)
 *         └───────────────────────────────────────────────────── UTC time
 */
std::optional<FixData> parseRMC(const std::vector<std::string_view>& f, FixData& fix) {
    if (f.size() < 12 || f[2] != "A") {
        return std::nullopt;
    }
    fix.timestamp = isoTimestamp(f[1], f[9]);
    fix.latitude  = parseCoord(f[3], f[4]);
    fix.latHemi   = std::string(f[4]);
    fix.longitude = parseCoord(f[5], f[6]);
    fix.lonHemi   = std::string(f[6]);
    from_chars(f[7].data(), f[7].data() + f[7].size(), fix.speedKmh);
    from_chars(f[8].data(), f[8].data() + f[8].size(), fix.headingDeg);
    return fix;
}

std::vector<std::string_view> splitFields(std::string_view line) {
    std::vector<std::string_view> out;
    std::size_t                   start = 0;
    while (true) {
        auto end = line.find(',', start);
        out.push_back(line.substr(start, end - start));
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return out;
}

void sendUbxUpdateRate(int fd, int updatePeriod_ms) {
    // Set update rate via UBX binary protocol
    // UBX-CFG-RATE (0x06 0x08), 6 payload bytes: measRate, navRate, timeRef
    uint8_t msg[] = {
        0xB5, 0x62,                                                                                                    // Sync chars
        0x06, 0x08,                                                                                                    // Class, ID
        0x06, 0x00,                                                                                                    // Length (6)
        static_cast<uint8_t>(updatePeriod_ms & 0xFF), static_cast<uint8_t>((updatePeriod_ms >> 8) & 0xFF), 0x01, 0x00, // navRate = 1
        0x01, 0x00,                                                                                                    // timeRef = UTC
        0x00, 0x00                                                                                                     // Checksum (to be filled)
    };
    uint8_t ckA = 0, ckB = 0;
    for (int i = 2; i < 12; ++i) {
        ckA += msg[i];
        ckB += ckA;
    }
    msg[12] = ckA;
    msg[13] = ckB;
    write(fd, msg, sizeof(msg));
}

void sendGpsUpdateRateCommand(int fd, int updatePeriod_ms = 1000) {
    // Send model-specific update rate commands
    sendUbxUpdateRate(fd, updatePeriod_ms); // 5 Hz for UBX

    std::string pmtk_5hz = std::format("$PMTK220,{}*2C\r\n", updatePeriod_ms);
    write(fd, pmtk_5hz.c_str(), pmtk_5hz.length());

    std::println("[INFO] Sent update-rate commands for UBX and PMTK");

    const char* pmtk_query_model = "$PMTK605*31\r\n"; // Query MTK firmware version
    const char* pubx_query_model = "$PUBX,00*33\r\n"; // Query u-blox info
    write(fd, pmtk_query_model, strlen(pmtk_query_model));
    write(fd, pubx_query_model, strlen(pubx_query_model));
    std::println("[INFO] Sent GNSS model/version query commands.");
}

extern "C" void EMSCRIPTEN_KEEPALIVE onGpsLineReceived(const char* cstr) {
    static FixData   fix;
    std::string_view line(cstr);
    if (!line.empty() && line.back() == '\r') {
        line.remove_suffix(1);
    }

    if (!line.starts_with("$")) {
        return;
    }

    auto fields = splitFields(line);
    if (line.starts_with("$GPRMC")) {
        if (auto f = parseRMC(fields, fix)) {
            printFix(fix);
            writeGpx(fix);
        }
    } else if (line.starts_with("$GPGGA")) {
        parseGGA(fields, fix);
    } else if (line.starts_with("$GPVTG")) {
        parseVTG(fields, fix);
    } else if (line.starts_with("$GPGSA")) {
        parseGSA(fields, fix);
    } else if (line.starts_with("$GPGLL")) {
        parseGLL(fields, fix);
    } else if (line.starts_with("$GPGSV")) {
        // parseGSV(fields); // optional
    } else if (line.starts_with("$GPTXT")) {
        std::println("[TEXT] {}", line);
    } else {
        std::println("[INFO] Unhandled NMEA: {}", line);
    }
}

#if !defined(__EMSCRIPTEN__)
int openSerial(std::string_view device, int baud = B9600) {
    int fd = ::open(device.data(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        throw std::runtime_error("Can't open serial port");
    }
    termios tio{};
    tcgetattr(fd, &tio);
    cfsetispeed(&tio, baud);
    cfsetospeed(&tio, baud);
    tio.c_cflag |= (CLOCAL | CREAD);
    tio.c_cflag &= ~CSIZE;
    tio.c_cflag |= CS8;
    tio.c_cflag &= ~(PARENB | CSTOPB);
    tcsetattr(fd, TCSANOW, &tio);
    return fd;
}

int main(int argc, char** argv) {
    const char* device = (argc > 1) ? argv[1] : "/dev/ttyACM1";
    int         fd     = openSerial(device);
    sendGpsUpdateRateCommand(fd);
    std::string buffer;
    char        temp[256];

    while (true) {
        pollfd pfd = {fd, POLLIN, 0};
        if (poll(&pfd, 1, 500) > 0) {
            ssize_t n = read(fd, temp, sizeof(temp));
            if (n > 0) {
                buffer.append(temp, n);
            }

            std::size_t pos;
            while ((pos = buffer.find('\n')) != std::string::npos) {
                auto line = buffer.substr(0, pos);
                buffer.erase(0, pos + 1);
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                onGpsLineReceived(line.c_str());
            }
        }
    }
    return 0;
}
#else
extern "C" EMSCRIPTEN_KEEPALIVE void setBaudRate(int baud);
extern "C" EMSCRIPTEN_KEEPALIVE void connectSerial();
extern "C" EMSCRIPTEN_KEEPALIVE void disconnectSerial();

namespace {
    int desiredBaudRate = 9600;
}

extern "C" EMSCRIPTEN_KEEPALIVE void setBaudRate(int baud) {
    desiredBaudRate = baud;
}

EM_JS(void, connectSerial, (), {
    if (!window.__gps_port) {
        navigator.serial.requestPort().then(openPort).catch(console.error);
    } else {
        openPort(window.__gps_port);
    }
});

EM_JS(void, disconnectSerial, (), {
    if (window.__gps_reader) {
        window.__gps_reader.cancel();
        window.__gps_reader = null;
    }
    if (window.__gps_port) {
        window.__gps_port.close().catch(console.error);
        window.__gps_port = null;
    }
    console.log("WASM: Serial disconnected.");
});

EM_JS(void, start_webserial, (), {
    if (!("serial" in navigator)) {
        document.body.innerHTML = "<p>Your browser does not support WebSerial. Please use Chrome or Edge on desktop.</p>";
        return;
    }

    console.log("WASM: Awaiting WebSerial input from JavaScript.");

    async function openPort(port) {
        window.__gps_port = port;
        const baud = Module._desiredBaudRate || 9600;
        console.log("WASM: Opening serial port at", baud);
        await port.open({ baudRate: baud });

        const decoder = new TextDecoderStream();
        const inputDone = port.readable.pipeTo(decoder.writable);
        const inputStream = decoder.readable;
        const reader = inputStream.getReader();
        window.__gps_reader = reader;

        let buffer = "";
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            if (value) buffer += value;

            let lineEnd;
            while ((lineEnd = buffer.indexOf("\n")) >= 0) {
                const line = buffer.slice(0, lineEnd).trimEnd();
                buffer = buffer.slice(lineEnd + 1);
                if (line.startsWith("$") && line.length > 6) {
                    const type = line.substring(3, 6);
                    switch (type) {
                        case "RMC":
                        case "GGA":
                        case "VTG":
                        case "GSA":
                        case "GLL":
                        case "GSV":
                        case "TXT":
                            Module.ccall("onGpsLineReceived", null, ["string"], [line]);
                            break;
                        default:
                            console.log("[INFO] Unhandled NMEA:", line);
                    }
                }
            }
        }
    }

    navigator.serial.getPorts().then(async ports => {
        if (ports.length > 0) {
            console.log("WASM: Found previously granted serial port.");
            await openPort(ports[0]);
        }
    });

    const connectBtn = document.createElement('button');
    connectBtn.textContent = "Connect to GPS (Serial)";
    connectBtn.style = "font-size: 1.5em; margin-top: 2em;";
    document.body.appendChild(connectBtn);

    connectBtn.addEventListener('click', async () => {
        connectBtn.disabled = true;
        try {
            const port = await navigator.serial.requestPort();
            await openPort(port);
        } catch (e) {
            console.error("WebSerial error:", e);
        }
    });

    // Optional auto-reconnect every 5 seconds
    setInterval(() => {
        if (!window.__gps_port) {
            navigator.serial.getPorts().then(ports => {
                if (ports.length > 0) {
                    console.log("WASM: Auto-reconnecting...");
                    openPort(ports[0]);
                }
            });
        }
    }, 5000);
});

int main() {
    start_webserial();
    return 0;
}

#endif
