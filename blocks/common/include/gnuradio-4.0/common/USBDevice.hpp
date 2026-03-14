#ifndef GNURADIO_USB_DEVICE_HPP
#define GNURADIO_USB_DEVICE_HPP

#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/usbdevice_fs.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace gr::blocks::common {

struct USBDeviceId {
    std::uint16_t    vendorId;
    std::uint16_t    productId;
    std::string_view description;
};

struct USBDeviceInfo {
    std::uint16_t vendorId  = 0;
    std::uint16_t productId = 0;
    std::uint8_t  busNum    = 0;
    std::uint8_t  devNum    = 0;
    std::string   devPath; // "/dev/bus/usb/001/004"
    std::string   product;
    std::string   manufacturer;
    bool          accessible = false;
};

#if defined(__linux__)

namespace detail {

inline std::string readSysfsAttr(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::string   content;
    if (file && std::getline(file, content)) {
        while (!content.empty() && (content.back() == '\n' || content.back() == '\r' || content.back() == ' ')) {
            content.pop_back();
        }
    }
    return content;
}

inline std::uint16_t parseHex16(std::string_view s) {
    std::uint16_t result = 0;
    std::from_chars(s.data(), s.data() + s.size(), result, 16);
    return result;
}

inline std::uint8_t parseDec8(std::string_view s) {
    std::uint8_t result = 0;
    std::from_chars(s.data(), s.data() + s.size(), result);
    return result;
}

} // namespace detail

[[nodiscard]] inline bool canAccessUSBDevice(const USBDeviceInfo& device) { return ::access(device.devPath.c_str(), R_OK | W_OK) == 0; }

inline std::vector<USBDeviceInfo> enumerateUSBDevices(std::span<const USBDeviceId> vidPidFilter = {}) {
    namespace fs = std::filesystem;

    std::vector<USBDeviceInfo> result;
    std::error_code            ec;

    for (const auto& entry : fs::directory_iterator("/sys/bus/usb/devices", ec)) {
        auto idVendorPath = entry.path() / "idVendor";
        if (!fs::exists(idVendorPath, ec)) {
            continue;
        }

        auto vid = detail::parseHex16(detail::readSysfsAttr(idVendorPath));
        auto pid = detail::parseHex16(detail::readSysfsAttr(entry.path() / "idProduct"));

        if (!vidPidFilter.empty()) {
            bool match = false;
            for (const auto& f : vidPidFilter) {
                if (f.vendorId == vid && f.productId == pid) {
                    match = true;
                    break;
                }
            }
            if (!match) {
                continue;
            }
        }

        auto busNum = detail::parseDec8(detail::readSysfsAttr(entry.path() / "busnum"));
        auto devNum = detail::parseDec8(detail::readSysfsAttr(entry.path() / "devnum"));

        USBDeviceInfo info;
        info.devPath      = std::format("/dev/bus/usb/{:03d}/{:03d}", busNum, devNum);
        info.product      = detail::readSysfsAttr(entry.path() / "product");
        info.manufacturer = detail::readSysfsAttr(entry.path() / "manufacturer");
        info.vendorId     = vid;
        info.productId    = pid;
        info.busNum       = busNum;
        info.devNum       = devNum;
        info.accessible   = canAccessUSBDevice(info);
        result.push_back(std::move(info));
    }
    return result;
}

struct USBDevice {
    using Result = std::expected<void, std::string>;

    int _fd        = -1;
    int _interface = -1;

    USBDevice()                            = default;
    USBDevice(const USBDevice&)            = delete;
    USBDevice& operator=(const USBDevice&) = delete;
    USBDevice(USBDevice&& o) noexcept : _fd(std::exchange(o._fd, -1)), _interface(std::exchange(o._interface, -1)) {}
    USBDevice& operator=(USBDevice&& o) noexcept {
        close();
        _fd        = std::exchange(o._fd, -1);
        _interface = std::exchange(o._interface, -1);
        return *this;
    }
    ~USBDevice() { close(); }

    [[nodiscard]] bool isOpen() const { return _fd >= 0; }

    // ── lifecycle ───────────────────────────────────────────────────────────

    [[nodiscard]] Result open(const USBDeviceInfo& device, int interfaceNum = 0) {
        _fd = ::open(device.devPath.c_str(), O_RDWR);
        if (_fd < 0) {
            int err = errno;
            if (err == EACCES) {
                return std::unexpected(std::format("permission denied opening '{}'"
                                                   "\n  try: sudo chmod 666 {}"
                                                   "\n  or add a udev rule:"
                                                   "\n    echo 'SUBSYSTEM==\"usb\", ATTR{{idVendor}}==\"{:04x}\", ATTR{{idProduct}}==\"{:04x}\", MODE=\"0666\"'"
                                                   "\n    | sudo tee /etc/udev/rules.d/99-usb-{:04x}-{:04x}.rules"
                                                   "\n    && sudo udevadm control --reload-rules && sudo udevadm trigger",
                    device.devPath, device.devPath, device.vendorId, device.productId, device.vendorId, device.productId));
            }
            return std::unexpected(std::format("failed to open '{}': {}", device.devPath, std::strerror(err)));
        }

        // detach kernel driver (e.g. dvb_usb_rtl28xxu) — ENODATA means none attached
        usbdevfs_disconnect_claim dc{};
        dc.interface = static_cast<unsigned>(interfaceNum);
        dc.flags     = USBDEVFS_DISCONNECT_CLAIM_EXCEPT_DRIVER;
        dc.driver[0] = '\0'; // disconnect any driver
        if (::ioctl(_fd, USBDEVFS_DISCONNECT_CLAIM, &dc) < 0) {
            // fallback: try separate disconnect + claim
            usbdevfs_ioctl cmd{};
            cmd.ifno          = interfaceNum;
            cmd.ioctl_code    = USBDEVFS_DISCONNECT;
            cmd.data          = nullptr;
            int disconnectRet = ::ioctl(_fd, USBDEVFS_IOCTL, &cmd);
            if (disconnectRet < 0 && errno != ENODATA) {
                // ENODATA = no driver attached — not an error
                int err = errno;
                ::close(_fd);
                _fd = -1;
                return std::unexpected(std::format("failed to detach kernel driver on interface {}: {}", interfaceNum, std::strerror(err)));
            }

            int iface = interfaceNum;
            if (::ioctl(_fd, USBDEVFS_CLAIMINTERFACE, &iface) < 0) {
                int err = errno;
                ::close(_fd);
                _fd = -1;
                if (err == EBUSY) {
                    return std::unexpected(std::format("interface {} already claimed on '{}'"
                                                       "\n  check: lsof {}",
                        interfaceNum, device.devPath, device.devPath));
                }
                return std::unexpected(std::format("failed to claim interface {}: {}", interfaceNum, std::strerror(err)));
            }
        }

        _interface = interfaceNum;
        return {};
    }

    void close() {
        if (_fd < 0) {
            return;
        }
        if (_interface >= 0) {
            int iface = _interface;
            ::ioctl(_fd, USBDEVFS_RELEASEINTERFACE, &iface);
            // re-attach kernel driver so the device returns to normal
            usbdevfs_ioctl cmd{};
            cmd.ifno       = _interface;
            cmd.ioctl_code = USBDEVFS_CONNECT;
            cmd.data       = nullptr;
            ::ioctl(_fd, USBDEVFS_IOCTL, &cmd);
            _interface = -1;
        }
        ::close(_fd);
        _fd = -1;
    }

    // ── control transfers ───────────────────────────────────────────────────

    [[nodiscard]] Result controlOut(std::uint8_t bmRequestType, std::uint8_t bRequest, std::uint16_t wValue, std::uint16_t wIndex, std::span<const std::uint8_t> data, unsigned timeoutMs = 300) {
        usbdevfs_ctrltransfer ctrl{};
        ctrl.bRequestType = bmRequestType;
        ctrl.bRequest     = bRequest;
        ctrl.wValue       = wValue;
        ctrl.wIndex       = wIndex;
        ctrl.wLength      = static_cast<std::uint16_t>(data.size());
        ctrl.timeout      = timeoutMs;
        ctrl.data         = const_cast<std::uint8_t*>(data.data()); // kernel reads, does not write

        if (::ioctl(_fd, USBDEVFS_CONTROL, &ctrl) < 0) {
            return std::unexpected(formatTransferError("control OUT", errno));
        }
        return {};
    }

    [[nodiscard]] std::expected<std::vector<std::uint8_t>, std::string> controlIn(std::uint8_t bmRequestType, std::uint8_t bRequest, std::uint16_t wValue, std::uint16_t wIndex, std::uint16_t length, unsigned timeoutMs = 300) {
        std::vector<std::uint8_t> buf(std::max<std::uint16_t>(length, 8));

        usbdevfs_ctrltransfer ctrl{};
        ctrl.bRequestType = bmRequestType;
        ctrl.bRequest     = bRequest;
        ctrl.wValue       = wValue;
        ctrl.wIndex       = wIndex;
        ctrl.wLength      = static_cast<std::uint16_t>(buf.size());
        ctrl.timeout      = timeoutMs;
        ctrl.data         = buf.data();

        int ret = ::ioctl(_fd, USBDEVFS_CONTROL, &ctrl);
        if (ret < 0) {
            return std::unexpected(formatTransferError("control IN", errno));
        }
        buf.resize(static_cast<std::size_t>(ret));
        return buf;
    }

    // ── bulk transfers ──────────────────────────────────────────────────────

    [[nodiscard]] std::expected<std::size_t, std::string> bulkRead(std::uint8_t endpoint, std::span<std::uint8_t> buf, unsigned timeoutMs = 100) {
        usbdevfs_bulktransfer bulk{};
        bulk.ep      = endpoint;
        bulk.len     = static_cast<unsigned>(buf.size());
        bulk.timeout = timeoutMs;
        bulk.data    = buf.data();

        int ret = ::ioctl(_fd, USBDEVFS_BULK, &bulk);
        if (ret < 0) {
            if (errno == ETIMEDOUT) {
                return 0UZ; // timeout is not an error for bulk reads — just no data yet
            }
            return std::unexpected(formatTransferError("bulk read", errno));
        }
        return static_cast<std::size_t>(ret);
    }

    [[nodiscard]] Result reset() {
        if (_fd < 0) {
            return std::unexpected(std::string("USBDevice: not open"));
        }
        if (::ioctl(_fd, USBDEVFS_RESET, nullptr) < 0) {
            return std::unexpected(std::format("USB reset failed: {}", std::strerror(errno)));
        }
        // after reset the interface claim is dropped — caller must re-claim + re-init
        _interface = -1;
        return {};
    }

    [[nodiscard]] Result clearHalt(std::uint8_t endpoint) {
        if (_fd < 0) {
            return std::unexpected(std::string("USBDevice: not open"));
        }
        unsigned ep = endpoint;
        if (::ioctl(_fd, USBDEVFS_CLEAR_HALT, &ep) < 0) {
            return std::unexpected(std::format("clear halt on endpoint 0x{:02X} failed: {}", endpoint, std::strerror(errno)));
        }
        return {};
    }

private:
    [[nodiscard]] static std::string formatTransferError(std::string_view op, int err) {
        std::string_view hint;
        switch (err) {
        case ENODEV: hint = " (device disconnected)"; break;
        case EPIPE: hint = " (transfer stalled — device rejected request)"; break;
        case ETIMEDOUT: hint = " (transfer timed out)"; break;
        case EBUSY: hint = " (endpoint busy)"; break;
        case EIO: hint = " (I/O error — possible hardware fault)"; break;
        default: hint = ""; break;
        }
        return std::format("{} failed: {}{}", op, std::strerror(err), hint);
    }
};

#else // !__linux__

// stub: non-Linux platforms use WebUSB shims or similar; permissions handled by the browser/OS
[[nodiscard]] inline bool         canAccessUSBDevice(const USBDeviceInfo&) { return true; }
inline std::vector<USBDeviceInfo> enumerateUSBDevices(std::span<const USBDeviceId> = {}) { return {}; }

struct USBDevice {
    using Result = std::expected<void, std::string>;

    [[nodiscard]] bool   isOpen() const { return false; }
    [[nodiscard]] Result open(const USBDeviceInfo&, int = 0) { return std::unexpected(std::string("USBDevice: Linux-only")); }
    void                 close() {}

    [[nodiscard]] Result controlOut(std::uint8_t, std::uint8_t, std::uint16_t, std::uint16_t, std::span<const std::uint8_t>, unsigned = 300) { return std::unexpected(std::string("USBDevice: Linux-only")); }

    [[nodiscard]] std::expected<std::vector<std::uint8_t>, std::string> controlIn(std::uint8_t, std::uint8_t, std::uint16_t, std::uint16_t, std::uint16_t, unsigned = 300) { return std::unexpected(std::string("USBDevice: Linux-only")); }

    [[nodiscard]] std::expected<std::size_t, std::string> bulkRead(std::uint8_t, std::span<std::uint8_t>, unsigned = 100) { return std::unexpected(std::string("USBDevice: Linux-only")); }

    [[nodiscard]] Result reset() { return std::unexpected(std::string("USBDevice: Linux-only")); }
    [[nodiscard]] Result clearHalt(std::uint8_t) { return std::unexpected(std::string("USBDevice: Linux-only")); }
};

#endif // __linux__

} // namespace gr::blocks::common

#endif // GNURADIO_USB_DEVICE_HPP
