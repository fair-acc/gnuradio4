#ifndef GNURADIO_NMEA_DEVICE_HPP
#define GNURADIO_NMEA_DEVICE_HPP

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gnuradio-4.0/common/DeviceRegistry.hpp>
#include <gnuradio-4.0/common/ScopedFd.hpp>
#include <gnuradio-4.0/common/USBDevice.hpp>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#include <emscripten/threading.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else // POSIX (Linux, macOS, FreeBSD, ...)
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace gr::timing {

enum class BaudRate : std::uint32_t { Baud4800 = 4800, Baud9600 = 9600, Baud19200 = 19200, Baud38400 = 38400, Baud57600 = 57600, Baud115200 = 115200, Baud230400 = 230400 };

using gr::blocks::common::USBDeviceId;

inline constexpr std::array kKnownGpsReceiverIds{
    USBDeviceId{0x1546, 0x01a7, "u-blox 7"},
    USBDeviceId{0x1546, 0x01a8, "u-blox 8"},
    USBDeviceId{0x1546, 0x01a9, "u-blox M8"},
    USBDeviceId{0x1546, 0x0502, "u-blox M9/M10"},
    USBDeviceId{0x1546, 0x01a6, "u-blox 6"},
    USBDeviceId{0x067b, 0x2303, "Prolific PL2303 (common GPS bridge)"},
    USBDeviceId{0x0681, 0x0002, "SiRF GPS"},
    USBDeviceId{0x1199, 0x0120, "Sierra Wireless GPS"},
    USBDeviceId{0x10c4, 0xea60, "CP210x (common GPS bridge)"},
    USBDeviceId{0x0403, 0x6001, "FTDI FT232R (common GPS bridge)"},
    USBDeviceId{0x0403, 0x6015, "FTDI FT-X (common GPS bridge)"},
    USBDeviceId{0x1a86, 0x7523, "CH340 (common GPS bridge)"},
};

struct NMEADeviceInfo {
    std::string   devicePath; // "/dev/ttyACM0", "COM3", or "webserial:0"
    std::string   model;      // "u-blox 7 - GPS/GNSS Receiver"
    std::string   vendor;     // "u-blox AG"
    std::uint16_t vendorId       = 0;
    std::uint16_t productId      = 0;
    bool          knownGpsDevice = false;
};

namespace detail {

inline bool isKnownGpsDevice(std::uint16_t vid, std::uint16_t pid) {
    return std::ranges::any_of(kKnownGpsReceiverIds, [vid, pid](const auto& entry) { return entry.vendorId == vid && entry.productId == pid; });
}

inline std::string lookupDescription(std::uint16_t vid, std::uint16_t pid) {
    auto it = std::ranges::find_if(kKnownGpsReceiverIds, [vid, pid](const auto& e) { return e.vendorId == vid && e.productId == pid; });
    return it != kKnownGpsReceiverIds.end() ? std::string(it->description) : std::string{};
}

inline bool isNMEALine(std::string_view line) { return line.size() >= 6 && line[0] == '$' && std::isalpha(static_cast<unsigned char>(line[1])) && std::isalpha(static_cast<unsigned char>(line[2])); }

// ── Linux ──────────────────────────────────────────────────────────────────────
#if defined(__linux__)

using gr::blocks::common::detail::parseHex16;
using gr::blocks::common::detail::readSysfsAttr;

inline std::optional<std::filesystem::path> findUSBDeviceDir(const std::filesystem::path& ttyDeviceDir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto            current = fs::canonical(ttyDeviceDir, ec);
    if (ec) {
        return std::nullopt;
    }

    for (int depth = 0; depth < 10; ++depth) {
        if (fs::exists(current / "idVendor", ec)) {
            return current;
        }
        auto parent = current.parent_path();
        if (parent == current) {
            break;
        }
        current = parent;
    }
    return std::nullopt;
}

inline std::optional<NMEADeviceInfo> readLinuxDeviceInfo(std::string_view ttyName) {
    namespace fs     = std::filesystem;
    fs::path sysPath = fs::path("/sys/class/tty") / ttyName / "device";

    std::error_code ec;
    if (!fs::exists(sysPath, ec)) {
        return std::nullopt;
    }

    auto usbDir = findUSBDeviceDir(sysPath);
    if (!usbDir) {
        return std::nullopt;
    }

    NMEADeviceInfo info;
    info.devicePath     = std::format("/dev/{}", ttyName);
    info.vendorId       = parseHex16(readSysfsAttr(*usbDir / "idVendor"));
    info.productId      = parseHex16(readSysfsAttr(*usbDir / "idProduct"));
    info.vendor         = readSysfsAttr(*usbDir / "manufacturer");
    info.model          = readSysfsAttr(*usbDir / "product");
    info.knownGpsDevice = isKnownGpsDevice(info.vendorId, info.productId);
    return info;
}

inline std::string findProcessHoldingDevice(std::string_view devicePath) {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto            deviceCanon = fs::canonical(devicePath, ec);
    if (ec) {
        return {};
    }

    for (const auto& procEntry : fs::directory_iterator("/proc", ec)) {
        auto pidName = procEntry.path().filename().string();
        if (pidName.empty() || !std::isdigit(static_cast<unsigned char>(pidName[0]))) {
            continue;
        }

        std::error_code fdEc;
        for (const auto& fdEntry : fs::directory_iterator(procEntry.path() / "fd", fdEc)) {
            auto target = fs::read_symlink(fdEntry.path(), fdEc);
            if (!fdEc && fs::canonical(target, fdEc) == deviceCanon) {
                std::string comm = readSysfsAttr(procEntry.path() / "comm");
                return std::format("PID {} ({})", pidName, comm.empty() ? "unknown" : comm);
            }
        }
    }
    return {};
}

inline std::vector<NMEADeviceInfo> enumerateSerialDevices(bool ignoreVidPidFilter) {
    namespace fs = std::filesystem;
    std::vector<NMEADeviceInfo> result;
    std::error_code             ec;

    fs::path ttyDir("/sys/class/tty");
    if (!fs::exists(ttyDir, ec)) {
        return result;
    }

    for (const auto& entry : fs::directory_iterator(ttyDir, ec)) {
        auto name = entry.path().filename().string();
        if (!name.starts_with("ttyACM") && !name.starts_with("ttyUSB")) {
            continue;
        }

        auto info = readLinuxDeviceInfo(name);
        if (!info) {
            continue;
        }

        if (ignoreVidPidFilter || info->knownGpsDevice) {
            result.push_back(std::move(*info));
        }
    }
    return result;
}

inline std::string noDevicePermissionHint() {
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator("/sys/class/tty", ec)) {
        auto name = entry.path().filename().string();
        if (name.starts_with("ttyACM") || name.starts_with("ttyUSB")) {
            return "serial devices found but could not read USB info."
                   "\nCheck permissions: run 'ls -la /dev/ttyACM* /dev/ttyUSB*'."
                   "\nIf permission denied, add your user to the 'dialout' group:"
                   "\n  sudo usermod -aG dialout $USER"
                   "\n  (log out and back in for the change to take effect)";
        }
    }
    return "no serial GPS device found. Connect a USB GPS receiver and try again.";
}

// ── macOS ──────────────────────────────────────────────────────────────────────
#elif defined(__APPLE__)

inline std::vector<NMEADeviceInfo> enumerateSerialDevices(bool ignoreVidPidFilter) {
    namespace fs = std::filesystem;
    std::vector<NMEADeviceInfo> result;
    std::error_code             ec;

    // macOS USB serial devices appear as /dev/tty.usbmodem* (CDC ACM) or /dev/tty.usbserial* (FTDI/CP210x/etc.)
    for (const auto& entry : fs::directory_iterator("/dev", ec)) {
        auto name = entry.path().filename().string();
        if (!name.starts_with("tty.usbmodem") && !name.starts_with("tty.usbserial") && !name.starts_with("cu.usbmodem") && !name.starts_with("cu.usbserial")) {
            continue;
        }

        NMEADeviceInfo info;
        info.devicePath = entry.path().string();
        // VID/PID not available without IOKit — would require Objective-C bridge
        // best-effort: mark all USB serial devices as candidates
        info.knownGpsDevice = ignoreVidPidFilter;

        if (ignoreVidPidFilter) {
            result.push_back(std::move(info));
        }
    }
    return result;
}

inline std::string findProcessHoldingDevice([[maybe_unused]] std::string_view devicePath) {
    // macOS: flock() provides lock detection; no /proc equivalent
    return {};
}

inline std::string noDevicePermissionHint() {
    return "no serial GPS device found. Connect a USB GPS receiver and try again."
           "\nOn macOS, check System Settings > Privacy & Security > Serial Port access.";
}

// ── Windows ────────────────────────────────────────────────────────────────────
#elif defined(_WIN32)

inline std::vector<NMEADeviceInfo> enumerateSerialDevices(bool ignoreVidPidFilter) {
    std::vector<NMEADeviceInfo> result;

    // enumerate COM ports from the registry
    HKEY hKey = nullptr;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DEVICEMAP\\SERIALCOMM", 0, KEY_READ, &hKey) != ERROR_SUCCESS) {
        return result;
    }

    std::array<char, 256> valueName{};
    std::array<char, 256> valueData{};
    for (DWORD i = 0;; ++i) {
        DWORD nameLen = static_cast<DWORD>(valueName.size());
        DWORD dataLen = static_cast<DWORD>(valueData.size());
        DWORD type    = 0;

        if (RegEnumValueA(hKey, i, valueName.data(), &nameLen, nullptr, &type, reinterpret_cast<LPBYTE>(valueData.data()), &dataLen) != ERROR_SUCCESS) {
            break;
        }
        if (type != REG_SZ) {
            continue;
        }

        NMEADeviceInfo info;
        info.devicePath = std::string(valueData.data(), dataLen > 0 ? dataLen - 1 : 0); // strip null terminator
        // VID/PID: would require SetupAPI (SetupDiGetClassDevs + SetupDiGetDeviceInstanceId)
        // best-effort: include all COM ports when ignoreVidPidFilter is set
        info.knownGpsDevice = ignoreVidPidFilter;

        if (ignoreVidPidFilter) {
            result.push_back(std::move(info));
        }
    }
    RegCloseKey(hKey);
    return result;
}

inline std::string findProcessHoldingDevice([[maybe_unused]] std::string_view devicePath) {
    // Windows: exclusive access is enforced by CreateFile (dwShareMode=0)
    return {};
}

inline std::string noDevicePermissionHint() { return "no serial GPS device found. Connect a USB GPS receiver and check Device Manager for COM ports."; }

#endif // platform

// ── POSIX shared (Linux + macOS) ───────────────────────────────────────────────
#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)

inline constexpr speed_t toTermiosSpeed(BaudRate rate) {
    switch (rate) {
    case BaudRate::Baud4800: return B4800;
    case BaudRate::Baud9600: return B9600;
    case BaudRate::Baud19200: return B19200;
    case BaudRate::Baud38400: return B38400;
    case BaudRate::Baud57600: return B57600;
    case BaudRate::Baud115200: return B115200;
#if defined(B230400) // not defined on all POSIX systems
    case BaudRate::Baud230400: return B230400;
#else
    case BaudRate::Baud230400: return B115200; // fallback
#endif
    }
    return B9600;
}

#endif // POSIX

} // namespace detail

// ── WASM: WebSerialDevice + SPSC byte queue + JS bridge ─────────────────────────
#if defined(__EMSCRIPTEN__)

namespace detail {

struct SpscByteQueue {
    static constexpr std::size_t kCapacity = 16384;
    static constexpr std::size_t kMask     = kCapacity - 1;
    static_assert((kCapacity & kMask) == 0, "capacity must be a power of 2");

    std::array<char, kCapacity> _buf{};
    std::atomic<std::size_t>    _head{0};
    std::atomic<std::size_t>    _tail{0};
    std::atomic<bool>           portOpen{false};
    std::atomic<bool>           closed{false};

    std::size_t push(const char* src, std::size_t len) {
        const auto head = _head.load(std::memory_order_relaxed);
        const auto tail = _tail.load(std::memory_order_acquire);
        const auto free = kCapacity - (head - tail);
        const auto n    = std::min(len, free);
        for (std::size_t i = 0; i < n; ++i) {
            _buf[(head + i) & kMask] = src[i];
        }
        _head.store(head + n, std::memory_order_release);
        return n;
    }

    std::size_t pop(char* dst, std::size_t maxLen) {
        const auto tail      = _tail.load(std::memory_order_relaxed);
        const auto head      = _head.load(std::memory_order_acquire);
        const auto available = head - tail;
        const auto n         = std::min(maxLen, available);
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = _buf[(tail + i) & kMask];
        }
        _tail.store(tail + n, std::memory_order_release);
        return n;
    }

    void reset() {
        _head.store(0, std::memory_order_relaxed);
        _tail.store(0, std::memory_order_relaxed);
        portOpen.store(false, std::memory_order_relaxed);
        closed.store(false, std::memory_order_relaxed);
    }
};

} // namespace detail

inline constexpr std::size_t                                 kMaxWebSerialPorts = 4;
inline std::array<detail::SpscByteQueue, kMaxWebSerialPorts> webSerialQueues;

// clang-format off
EM_JS(void, js_registerSerialDevice, (), {
    var reg = window.__gr_devices = window.__gr_devices || {};
    reg["serial"] = {
        name: "Serial Port (GPS/GNSS)",
        available: !!(navigator && navigator.serial),
        grantedCount: 0,
        config: { baudRate: 9600 },
        request: async function(baudRate, useFilters) {
            if (!navigator || !navigator.serial) {
                throw new Error("WebSerial API not available \u2014 use Chrome or Edge");
            }
            baudRate = baudRate || this.config.baudRate || 9600;

            var options = {};
            if (useFilters) {
                var count = Module._knownGpsDeviceCount();
                var filters = [];
                for (var i = 0; i < count; i++) {
                    filters.push({
                        usbVendorId: Module._knownGpsVendorId(i),
                        usbProductId: Module._knownGpsProductId(i)
                    });
                }
                options.filters = filters;
            }
            var port = await navigator.serial.requestPort(options);
            await port.open({ baudRate: baudRate });

            var ws = window.__gr_webserial = window.__gr_webserial || { ports: [] };
            var idx = ws.ports.length;
            var reader = port.readable.getReader();
            ws.ports.push({ port: port, reader: reader, isOpen: true });

            window.__nmea_ports = ws.ports.map(function(e) { return e.port; });

            this.grantedCount++;
            Module._webserial_portOpened(idx);

            var self = this;
            (async function() {
                try {
                    while (true) {
                        var result = await reader.read();
                        if (result.done) {
                            break;
                        }
                        Module.ccall('webserial_onData', null,
                            ['number', 'array', 'number'],
                            [idx, result.value, result.value.length]);
                    }
                } catch (e) {
                    if (e.name !== 'AbortError') {
                        console.error('[WebSerial] read error:', e.message);
                    }
                }
                ws.ports[idx].isOpen = false;
                self.grantedCount = Math.max(0, self.grantedCount - 1);
                Module._webserial_portClosed(idx);
            })();
        }
    };
});

EM_JS(void, js_webserialRequestAndOpen, (int baudRate, int useFilters), {
    var dev = (window.__gr_devices || {})["serial"];
    if (dev && dev.request) {
        dev.request(baudRate, !!useFilters);
    } else {
        console.error('[WebSerial] serial device not registered \u2014 call DeviceRegistry::init() first');
    }
});

EM_JS(void, js_webserialClose, (int portIndex), {
    var state = (window.__gr_webserial || {}).ports || [];
    if (portIndex < 0 || portIndex >= state.length || !state[portIndex]) {
        return;
    }
    var entry = state[portIndex];
    if (!entry.isOpen) {
        return;
    }
    entry.isOpen = false;
    (async function() {
        try {
            if (entry.reader) {
                await entry.reader.cancel();
                entry.reader.releaseLock();
            }
            await entry.port.close();
        } catch (e) {
            console.error('[WebSerial] close:', e.message);
        }
    })();
});

EM_JS(int, js_checkSerialApiAvailable, (), { return !!(navigator && navigator.serial) ? 1 : 0; });

EM_JS(int, js_getGrantedPortCount, (), { return (window.__nmea_ports || []).length; });

EM_JS(int, js_getPortVendorId, (int index), {
    var ports = window.__nmea_ports || [];
    if (index < 0 || index >= ports.length) {
        return 0;
    }
    var info = ports[index].getInfo();
    return info.usbVendorId || 0;
});

EM_JS(int, js_getPortProductId, (int index), {
    var ports = window.__nmea_ports || [];
    if (index < 0 || index >= ports.length) {
        return 0;
    }
    var info = ports[index].getInfo();
    return info.usbProductId || 0;
});

EM_ASYNC_JS(int, js_webserialWrite, (int portIndex, const uint8_t* dataPtr, int dataLen), {
    var ws = window.__gr_webserial;
    if (!ws || portIndex < 0 || portIndex >= ws.ports.length) return -1;
    var entry = ws.ports[portIndex];
    if (!entry || !entry.isOpen || !entry.port || !entry.port.writable) return -1;
    try {
        var data = new Uint8Array(dataLen);
        for (var i = 0; i < dataLen; i++) data[i] = HEAPU8[dataPtr + i];
        var writer = entry.port.writable.getWriter();
        await writer.write(data);
        writer.releaseLock();
        return dataLen;
    } catch (e) {
        console.error('[WebSerial] write error:', e.message);
        return -1;
    }
});
// clang-format on

/// DeviceBase implementation for the browser WebSerial API. Self-registers as "serial".
/// JS bridge code (EM_JS) handles requestPort/open/read; data arrives via SpscByteQueue.
struct WebSerialDevice : gr::blocks::common::DeviceBase {
    std::atomic<bool> _apiAvailable{false};
    std::atomic<int>  _grantedCount{0};
    std::atomic<int>  _configuredBaudRate{9600};
    std::atomic<bool> _useDeviceFilters{true}; // filter WebSerial picker to known GPS VID/PIDs (suppresses ttyS0 etc.)
    std::string       _lastError;

    [[nodiscard]] std::string_view id() const noexcept override { return "serial"; }
    [[nodiscard]] std::string_view displayName() const noexcept override { return "Serial Port (GPS/GNSS)"; }

    void init() override {
        js_registerSerialDevice();
        _apiAvailable.store(js_checkSerialApiAvailable() != 0, std::memory_order_release);
    }

    [[nodiscard]] bool isApiAvailable() const noexcept override { return _apiAvailable.load(std::memory_order_acquire); }
    [[nodiscard]] int  grantedCount() const noexcept override { return _grantedCount.load(std::memory_order_acquire); }

    void configureBaudRate(BaudRate rate) { _configuredBaudRate.store(static_cast<int>(rate), std::memory_order_relaxed); }
    void setUseDeviceFilters(bool enable) { _useDeviceFilters.store(enable, std::memory_order_relaxed); }

    void requestPermission() override { js_webserialRequestAndOpen(_configuredBaudRate.load(std::memory_order_relaxed), _useDeviceFilters.load(std::memory_order_relaxed) ? 1 : 0); }

    [[nodiscard]] std::expected<int, std::string> connect(int portIndex, int /*baudRate*/) override {
        if (portIndex < 0 || static_cast<std::size_t>(portIndex) >= kMaxWebSerialPorts) {
            _lastError = std::format("invalid WebSerial port index {}", portIndex);
            return std::unexpected(_lastError);
        }
        if (!webSerialQueues[static_cast<std::size_t>(portIndex)].portOpen.load(std::memory_order_acquire)) {
            _lastError = std::format("WebSerial port {} not open — call requestPermission() from a user gesture first", portIndex);
            return std::unexpected(_lastError);
        }
        _lastError.clear();
        return portIndex;
    }

    void disconnect(int handle) override {
        if (handle < 0 || static_cast<std::size_t>(handle) >= kMaxWebSerialPorts) {
            return;
        }
        webSerialQueues[static_cast<std::size_t>(handle)].closed.store(true, std::memory_order_release);
    }

    [[nodiscard]] std::string lastError() const override { return _lastError; }

    [[nodiscard]] detail::SpscByteQueue& queue(int handle) { return webSerialQueues[static_cast<std::size_t>(handle)]; }

    void portOpened(int portIndex) {
        if (portIndex < 0 || static_cast<std::size_t>(portIndex) >= kMaxWebSerialPorts) {
            return;
        }
        webSerialQueues[static_cast<std::size_t>(portIndex)].portOpen.store(true, std::memory_order_release);
        _grantedCount.fetch_add(1, std::memory_order_release);
    }

    void portClosed(int portIndex) {
        if (portIndex < 0 || static_cast<std::size_t>(portIndex) >= kMaxWebSerialPorts) {
            return;
        }
        webSerialQueues[static_cast<std::size_t>(portIndex)].closed.store(true, std::memory_order_release);
        _grantedCount.fetch_sub(1, std::memory_order_release);
    }
};

extern "C" {

EMSCRIPTEN_KEEPALIVE inline void webserial_onData(int portIndex, const char* data, int len) {
    if (portIndex < 0 || static_cast<std::size_t>(portIndex) >= kMaxWebSerialPorts) {
        return;
    }
    webSerialQueues[static_cast<std::size_t>(portIndex)].push(data, static_cast<std::size_t>(len));
}

EMSCRIPTEN_KEEPALIVE inline void webserial_portOpened(int portIndex) {
    if (auto* dev = gr::blocks::common::DeviceRegistry::instance().findAs<WebSerialDevice>("serial")) {
        dev->portOpened(portIndex);
    }
}

EMSCRIPTEN_KEEPALIVE inline void webserial_portClosed(int portIndex) {
    if (auto* dev = gr::blocks::common::DeviceRegistry::instance().findAs<WebSerialDevice>("serial")) {
        dev->portClosed(portIndex);
    }
}

EMSCRIPTEN_KEEPALIVE inline int knownGpsDeviceCount() { return static_cast<int>(kKnownGpsReceiverIds.size()); }
EMSCRIPTEN_KEEPALIVE inline int knownGpsVendorId(int i) { return (i >= 0 && static_cast<std::size_t>(i) < kKnownGpsReceiverIds.size()) ? kKnownGpsReceiverIds[static_cast<std::size_t>(i)].vendorId : 0; }
EMSCRIPTEN_KEEPALIVE inline int knownGpsProductId(int i) { return (i >= 0 && static_cast<std::size_t>(i) < kKnownGpsReceiverIds.size()) ? kKnownGpsReceiverIds[static_cast<std::size_t>(i)].productId : 0; }

} // extern "C"

inline gr::blocks::common::AutoRegister autoRegWebSerial(std::make_shared<WebSerialDevice>());

#endif // __EMSCRIPTEN__

// ── Device discovery ────────────────────────────────────────────────────────────

inline std::vector<NMEADeviceInfo> discoverNMEADevices(bool ignoreVidPidFilter = false) {
    std::vector<NMEADeviceInfo> result;

#if defined(__EMSCRIPTEN__)
    int portCount = js_getGrantedPortCount();
    for (int i = 0; i < portCount; ++i) {
        NMEADeviceInfo info;
        info.devicePath     = std::format("webserial:{}", i);
        info.vendorId       = static_cast<std::uint16_t>(js_getPortVendorId(i));
        info.productId      = static_cast<std::uint16_t>(js_getPortProductId(i));
        info.knownGpsDevice = detail::isKnownGpsDevice(info.vendorId, info.productId);
        info.model          = detail::lookupDescription(info.vendorId, info.productId);

        if (ignoreVidPidFilter || info.knownGpsDevice) {
            result.push_back(std::move(info));
        }
    }
#else
    result = detail::enumerateSerialDevices(ignoreVidPidFilter);
#endif

    std::ranges::sort(result, [](const NMEADeviceInfo& a, const NMEADeviceInfo& b) {
        if (a.knownGpsDevice != b.knownGpsDevice) {
            return a.knownGpsDevice;
        }
        return a.devicePath < b.devicePath;
    });
    return result;
}

inline std::expected<NMEADeviceInfo, std::string> selectNMEADevice(std::string_view devicePath = {}, bool ignoreVidPidFilter = false) {

    if (!devicePath.empty()) {
#if defined(__EMSCRIPTEN__)
        if (!devicePath.starts_with("webserial:")) {
            return std::unexpected(std::format("invalid WASM device path '{}' (expected 'webserial:N')", devicePath));
        }
        auto devices = discoverNMEADevices(true);
        auto it      = std::ranges::find_if(devices, [&](const auto& d) { return d.devicePath == devicePath; });
        if (it != devices.end()) {
            return *it;
        }
        return std::unexpected(std::format("WebSerial device '{}' not found among granted ports", devicePath));
#elif defined(_WIN32)
        // on Windows, trust the user-provided COM port path
        NMEADeviceInfo info;
        info.devicePath = std::string(devicePath);
        return info;
#else
        namespace fs = std::filesystem;
        std::error_code ec;
        if (!fs::exists(devicePath, ec)) {
            return std::unexpected(std::format("device '{}' does not exist", devicePath));
        }
        // try to read USB metadata (Linux); on macOS this returns nullopt
#if defined(__linux__)
        auto name = fs::path(devicePath).filename().string();
        if (auto info = detail::readLinuxDeviceInfo(name)) {
            return *info;
        }
#endif
        NMEADeviceInfo fallback;
        fallback.devicePath = std::string(devicePath);
        return fallback;
#endif
    }

    // auto-detect
    auto devices = discoverNMEADevices(ignoreVidPidFilter);
    if (devices.empty()) {
        auto allDevices = discoverNMEADevices(true);
        if (!allDevices.empty()) {
            std::string deviceList;
            for (const auto& d : allDevices) {
                deviceList += std::format("\n  {} (VID:{:04x} PID:{:04x} - {})", d.devicePath, d.vendorId, d.productId, d.model.empty() ? "unknown" : d.model);
            }
            return std::unexpected(std::format("no known GPS device found, but {} serial device(s) detected:{}."
                                               "\nUse ignore_vid_pid_filter=true to try these, or add the VID:PID to kKnownGpsReceiverIds.",
                allDevices.size(), deviceList));
        }

#if defined(__EMSCRIPTEN__)
        return std::unexpected("no WebSerial GPS device found. Click 'Connect' to grant access to a serial port,"
                               " or check that your browser supports WebSerial (Chrome/Edge required).");
#else
        return std::unexpected(detail::noDevicePermissionHint());
#endif
    }

    return devices.front();
}

// thread-safe auto-detection usable from worker threads (WASM: polls atomics instead of calling EM_JS)
inline std::expected<NMEADeviceInfo, std::string> autoDetectNMEADevice() {
#if defined(__EMSCRIPTEN__)
    if (!gr::blocks::common::DeviceRegistry::instance().isGranted("serial")) {
        return std::unexpected(std::string("serial permission not granted"));
    }
    for (std::size_t i = 0; i < kMaxWebSerialPorts; ++i) {
        if (webSerialQueues[i].portOpen.load(std::memory_order_acquire)) {
            NMEADeviceInfo info;
            info.devicePath = std::format("webserial:{}", i);
            info.model      = std::format("WebSerial port {}", i);
            return info;
        }
    }
    return std::unexpected(std::string("no WebSerial port open — click 'Connect' to grant access"));
#else
    return selectNMEADevice({}, false);
#endif
}

inline void configureDefaultBaudRate([[maybe_unused]] BaudRate rate) {
#if defined(__EMSCRIPTEN__)
    if (auto* dev = gr::blocks::common::DeviceRegistry::instance().findAs<WebSerialDevice>("serial")) {
        dev->configureBaudRate(rate);
    }
#endif
}

// ── Serial port open (POSIX) ────────────────────────────────────────────────────
#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)

inline std::expected<std::intptr_t, std::string> openSerialPort(const std::string& devicePath, BaudRate baudRate = BaudRate::Baud9600) {

    int fd = ::open(devicePath.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        int err = errno;
        if (err == EACCES) {
#if defined(__linux__)
            return std::unexpected(std::format("permission denied opening '{}'. Add your user to the 'dialout' group:"
                                               "\n  sudo usermod -aG dialout $USER"
                                               "\n  (log out and back in for the change to take effect)",
                devicePath));
#else
            return std::unexpected(std::format("permission denied opening '{}'."
                                               "\nOn macOS, check System Settings > Privacy & Security.",
                devicePath));
#endif
        }
        return std::unexpected(std::format("failed to open '{}': {}", devicePath, std::strerror(err)));
    }

    gr::blocks::common::ScopedFd guard(fd);

    // advisory lock — detect if another process already holds the device
    if (::flock(guard.fd, LOCK_EX | LOCK_NB) < 0) {
        std::string holder;
#if defined(__linux__)
        holder = detail::findProcessHoldingDevice(devicePath);
#endif
        return std::unexpected(std::format("device '{}' is already in use{}."
                                           "\nClose the other process or use a different device.",
            devicePath, holder.empty() ? "" : std::format(" by {}", holder)));
    }

    termios tio{};
    tcgetattr(guard.fd, &tio);
    speed_t speed = detail::toTermiosSpeed(baudRate);
    cfsetispeed(&tio, speed);
    cfsetospeed(&tio, speed);

    tio.c_cflag |= static_cast<tcflag_t>(CLOCAL | CREAD);
    tio.c_cflag &= static_cast<tcflag_t>(~CSIZE);
    tio.c_cflag |= static_cast<tcflag_t>(CS8);
    tio.c_cflag &= static_cast<tcflag_t>(~(PARENB | CSTOPB));

    tio.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO | ECHOE | ISIG));
    tio.c_iflag &= static_cast<tcflag_t>(~(IXON | IXOFF | IXANY | ICRNL | INLCR));
    tio.c_oflag &= static_cast<tcflag_t>(~OPOST);

    tio.c_cc[VMIN]  = 0;
    tio.c_cc[VTIME] = 1; // 100ms read timeout

    if (tcsetattr(guard.fd, TCSANOW, &tio) < 0) {
        int err = errno;
        return std::unexpected(std::format("failed to configure '{}': {}", devicePath, std::strerror(err)));
    }
    tcflush(guard.fd, TCIOFLUSH);
    return guard.release();
}

inline std::expected<bool, std::string> probeForNMEA(const std::string& devicePath, BaudRate baudRate = BaudRate::Baud9600, std::chrono::milliseconds timeout = std::chrono::milliseconds{2000}) {
    using namespace std::chrono;

    auto fdResult = openSerialPort(devicePath, baudRate);
    if (!fdResult) {
        return std::unexpected(fdResult.error());
    }

    gr::blocks::common::ScopedFd guard(static_cast<int>(fdResult.value()));
    std::string                  buffer;
    std::array<char, 256UZ>      temp{};
    constexpr auto               kPollInterval = milliseconds{100};
    auto                         elapsed       = milliseconds::zero();

    while (elapsed < timeout) {
        pollfd pfd = {guard.fd, POLLIN, 0};
        int    ret = poll(&pfd, 1, static_cast<int>(kPollInterval.count()));
        if (ret > 0) {
            ssize_t n = ::read(guard.fd, temp.data(), temp.size());
            if (n > 0) {
                buffer.append(temp.data(), static_cast<std::size_t>(n));

                std::size_t pos;
                while ((pos = buffer.find('\n')) != std::string::npos) {
                    auto line = std::string_view(buffer).substr(0, pos);
                    if (!line.empty() && line.back() == '\r') {
                        line.remove_suffix(1);
                    }
                    if (detail::isNMEALine(line)) {
                        return true;
                    }
                    buffer.erase(0, pos + 1);
                }
            }
        }
        elapsed += kPollInterval;
    }
    return false;
}

// ── Serial port open (Windows) ──────────────────────────────────────────────────
#elif defined(_WIN32)

inline std::expected<std::intptr_t, std::string> openSerialPort(const std::string& devicePath, BaudRate baudRate = BaudRate::Baud9600) {

    // Windows COM ports > COM9 need the \\.\COMn prefix
    std::string winPath = devicePath;
    if (winPath.starts_with("COM") && !winPath.starts_with("\\\\.\\")) {
        winPath = "\\\\.\\" + winPath;
    }

    HANDLE hSerial = CreateFileA(winPath.c_str(), GENERIC_READ | GENERIC_WRITE,
        0, // exclusive access
        nullptr, OPEN_EXISTING, 0, nullptr);

    if (hSerial == INVALID_HANDLE_VALUE) {
        DWORD err = GetLastError();
        if (err == ERROR_ACCESS_DENIED) {
            return std::unexpected(std::format("device '{}' is already in use or access denied."
                                               "\nClose the other application or check Device Manager.",
                devicePath));
        }
        return std::unexpected(std::format("failed to open '{}' (error {})", devicePath, err));
    }

    DCB dcb{};
    dcb.DCBlength = sizeof(dcb);
    if (!GetCommState(hSerial, &dcb)) {
        CloseHandle(hSerial);
        return std::unexpected(std::format("failed to get serial state for '{}'", devicePath));
    }

    dcb.BaudRate    = static_cast<DWORD>(baudRate);
    dcb.ByteSize    = 8;
    dcb.StopBits    = ONESTOPBIT;
    dcb.Parity      = NOPARITY;
    dcb.fBinary     = TRUE;
    dcb.fDtrControl = DTR_CONTROL_ENABLE;

    if (!SetCommState(hSerial, &dcb)) {
        CloseHandle(hSerial);
        return std::unexpected(std::format("failed to configure '{}'", devicePath));
    }

    COMMTIMEOUTS timeouts{};
    timeouts.ReadIntervalTimeout         = MAXDWORD;
    timeouts.ReadTotalTimeoutMultiplier  = 0;
    timeouts.ReadTotalTimeoutConstant    = 100; // 100ms read timeout
    timeouts.WriteTotalTimeoutMultiplier = 0;
    timeouts.WriteTotalTimeoutConstant   = 100;
    SetCommTimeouts(hSerial, &timeouts);

    return reinterpret_cast<std::intptr_t>(hSerial);
}

inline std::expected<bool, std::string> probeForNMEA(const std::string& devicePath, BaudRate baudRate = BaudRate::Baud9600, std::chrono::milliseconds timeout = std::chrono::milliseconds{2000}) {
    using namespace std::chrono;

    auto handleResult = openSerialPort(devicePath, baudRate);
    if (!handleResult) {
        return std::unexpected(handleResult.error());
    }

    HANDLE                  hSerial = reinterpret_cast<HANDLE>(*handleResult);
    std::string             buffer;
    std::array<char, 256UZ> temp{};
    constexpr auto          kPollInterval = milliseconds{100};
    auto                    elapsed       = milliseconds::zero();

    while (elapsed < timeout) {
        DWORD bytesRead = 0;
        if (ReadFile(hSerial, temp.data(), static_cast<DWORD>(temp.size()), &bytesRead, nullptr) && bytesRead > 0) {
            buffer.append(temp.data(), bytesRead);

            std::size_t pos;
            while ((pos = buffer.find('\n')) != std::string::npos) {
                auto line = std::string_view(buffer).substr(0, pos);
                if (!line.empty() && line.back() == '\r') {
                    line.remove_suffix(1);
                }
                if (detail::isNMEALine(line)) {
                    CloseHandle(hSerial);
                    return true;
                }
                buffer.erase(0, pos + 1);
            }
        }
        Sleep(static_cast<DWORD>(kPollInterval.count()));
        elapsed += kPollInterval;
    }
    CloseHandle(hSerial);
    return false;
}

#endif // platform

// ── SerialPort — cross-platform serial I/O handle ───────────────────────────────

/// Cross-platform serial I/O handle wrapping POSIX fd, Win32 HANDLE, or WASM WebSerial index.
struct SerialPort {
    std::intptr_t _handle = -1; // POSIX fd, Win32 HANDLE, or WASM WebSerial port index

    SerialPort() = default;
    ~SerialPort() { close(); }

    SerialPort(const SerialPort&)            = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    SerialPort(SerialPort&& o) noexcept : _handle(std::exchange(o._handle, std::intptr_t{-1})) {}
    SerialPort& operator=(SerialPort&& o) noexcept {
        if (this != &o) {
            close();
            _handle = std::exchange(o._handle, std::intptr_t{-1});
        }
        return *this;
    }

    [[nodiscard]] static std::expected<SerialPort, std::string> open(const std::string& devicePath, [[maybe_unused]] BaudRate baudRate) {
        SerialPort port;

#if defined(__EMSCRIPTEN__)
        if (!devicePath.starts_with("webserial:")) {
            return std::unexpected(std::format("invalid WASM device path '{}' (expected 'webserial:N')", devicePath));
        }
        int  portIndex = -1;
        auto indexStr  = devicePath.substr(10);
        auto [ptr, ec] = std::from_chars(indexStr.data(), indexStr.data() + indexStr.size(), portIndex);
        if (ec != std::errc{} || portIndex < 0 || static_cast<std::size_t>(portIndex) >= kMaxWebSerialPorts) {
            return std::unexpected(std::format("invalid WebSerial port index in '{}'", devicePath));
        }
        auto* dev = gr::blocks::common::DeviceRegistry::instance().findAs<WebSerialDevice>("serial");
        if (!dev) {
            return std::unexpected("WebSerialDevice not registered");
        }
        auto result = dev->connect(portIndex, 0);
        if (!result) {
            return std::unexpected(result.error());
        }
        port._handle = *result;
#else
        auto result = openSerialPort(devicePath, baudRate);
        if (!result) {
            return std::unexpected(result.error());
        }
        port._handle = *result;
#endif
        return port;
    }

    void close() {
        if (_handle < 0) {
            return;
        }
#if defined(__EMSCRIPTEN__)
        if (auto* dev = gr::blocks::common::DeviceRegistry::instance().findAs<WebSerialDevice>("serial")) {
            dev->disconnect(static_cast<int>(_handle));
        }
#elif defined(_WIN32)
        CloseHandle(toWinHandle(_handle));
#else
        ::close(static_cast<int>(_handle));
#endif
        _handle = -1;
    }

    [[nodiscard]] std::size_t read(std::span<char> buffer, std::chrono::milliseconds timeout) {
        if (_handle < 0) {
            return 0;
        }
        auto timeoutMs = static_cast<int>(timeout.count());

#if defined(__EMSCRIPTEN__)
        auto* dev = gr::blocks::common::DeviceRegistry::instance().findAs<WebSerialDevice>("serial");
        if (!dev) {
            return 0;
        }
        auto& queue = dev->queue(static_cast<int>(_handle));
        auto  n     = queue.pop(buffer.data(), buffer.size());
        if (n > 0) {
            return n;
        }
        constexpr int kPollMs = 10;
        for (int elapsed = 0; elapsed < timeoutMs; elapsed += kPollMs) {
            if (queue.closed.load(std::memory_order_acquire)) {
                return 0;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
            n = queue.pop(buffer.data(), buffer.size());
            if (n > 0) {
                return n;
            }
        }
        return 0;

#elif defined(_WIN32)
        HANDLE       hSerial = toWinHandle(_handle);
        COMMTIMEOUTS timeouts{};
        timeouts.ReadIntervalTimeout        = MAXDWORD;
        timeouts.ReadTotalTimeoutMultiplier = 0;
        timeouts.ReadTotalTimeoutConstant   = static_cast<DWORD>(timeoutMs);
        SetCommTimeouts(hSerial, &timeouts);
        DWORD bytesRead = 0;
        if (ReadFile(hSerial, buffer.data(), static_cast<DWORD>(buffer.size()), &bytesRead, nullptr)) {
            return static_cast<std::size_t>(bytesRead);
        }
        return 0;

#else // POSIX
        int    posixFd = static_cast<int>(_handle);
        pollfd pfd     = {posixFd, POLLIN, 0};
        int    ret     = ::poll(&pfd, 1, timeoutMs);
        if (ret > 0) {
            ssize_t n = ::read(posixFd, buffer.data(), buffer.size());
            if (n > 0) {
                return static_cast<std::size_t>(n);
            }
        }
        return 0;
#endif
    }

    std::size_t write([[maybe_unused]] std::span<const std::uint8_t> data) {
        if (_handle < 0) {
            return 0;
        }
#if defined(__EMSCRIPTEN__)
        int ret = js_webserialWrite(static_cast<int>(_handle), reinterpret_cast<const uint8_t*>(data.data()), static_cast<int>(data.size()));
        return ret > 0 ? static_cast<std::size_t>(ret) : 0;

#elif defined(_WIN32)
        HANDLE hSerial      = toWinHandle(_handle);
        DWORD  bytesWritten = 0;
        if (WriteFile(hSerial, data.data(), static_cast<DWORD>(data.size()), &bytesWritten, nullptr)) {
            return static_cast<std::size_t>(bytesWritten);
        }
        return 0;

#else // POSIX
        ssize_t n = ::write(static_cast<int>(_handle), data.data(), data.size());
        return n >= 0 ? static_cast<std::size_t>(n) : 0;
#endif
    }

    [[nodiscard]] bool isOpen() const { return _handle >= 0; }

private:
#if defined(_WIN32)
    static HANDLE toWinHandle(std::intptr_t h) { return reinterpret_cast<HANDLE>(h); }
#endif
};

} // namespace gr::timing

#endif // GNURADIO_NMEA_DEVICE_HPP
