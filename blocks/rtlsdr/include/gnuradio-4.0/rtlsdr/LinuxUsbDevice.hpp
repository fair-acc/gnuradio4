#ifndef GNURADIO_LINUX_USB_DEVICE_HPP
#define GNURADIO_LINUX_USB_DEVICE_HPP

// Linux-native USB device access via /dev/bus/usb and sysfs.
// for a zero-dependency, MIT-licensed USB transport layer. Linux 2.6+ only.
//
// Reference documentation:
//   - linux/usbdevice_fs.h — ioctl definitions (USBDEVFS_CONTROL, USBDEVFS_BULK, etc.)
//     https://github.com/torvalds/linux/blob/master/include/uapi/linux/usbdevice_fs.h
//   - Documentation/usb/proc_usb_info.rst — /proc/bus/usb and /dev/bus/usb layout
//     https://docs.kernel.org/usb/proc_usb_info.html
//   - Documentation/driver-api/usb/URB.rst — USB request blocks (URB) and bulk transfers
//     https://docs.kernel.org/driver-api/usb/URB.html
//   - Documentation/driver-api/usb/usb.rst — general USB subsystem overview
//     https://docs.kernel.org/driver-api/usb/usb.html
//   - sysfs USB device attributes — /sys/bus/usb/devices/<busnum>-<devpath>/
//     idVendor, idProduct, product, manufacturer, busnum, devnum
//     https://docs.kernel.org/usb/proc_usb_info.html#the-sys-bus-usb-devices-directory

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/usbdevice_fs.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace gr::blocks::rtlsdr {

struct UsbDeviceInfo {
    std::string   devPath; // "/dev/bus/usb/001/004"
    std::string   product; // "RTL2838UHIDIR"
    std::uint16_t vendorId  = 0;
    std::uint16_t productId = 0;
    std::uint8_t  busNum    = 0;
    std::uint8_t  devNum    = 0;
};

struct LinuxUsbDevice {
    using Result = std::expected<void, std::string>;

    int _fd = -1;

    LinuxUsbDevice()                                 = default;
    LinuxUsbDevice(const LinuxUsbDevice&)            = delete;
    LinuxUsbDevice& operator=(const LinuxUsbDevice&) = delete;
    LinuxUsbDevice(LinuxUsbDevice&& o) noexcept : _fd(std::exchange(o._fd, -1)) {}
    LinuxUsbDevice& operator=(LinuxUsbDevice&& o) noexcept {
        close();
        _fd = std::exchange(o._fd, -1);
        return *this;
    }
    ~LinuxUsbDevice() { close(); }

    [[nodiscard]] bool isOpen() const { return _fd >= 0; }

    // ── device enumeration ──────────────────────────────────────────────────
    // scans /sys/bus/usb/devices/ for matching VID/PID pairs,
    // resolves busnum/devnum → /dev/bus/usb/BBB/DDD path

    [[nodiscard]] static std::expected<std::vector<UsbDeviceInfo>, std::string> enumerate(std::span<const std::pair<std::uint16_t, std::uint16_t>> vidPidFilter);

    // ── lifecycle ───────────────────────────────────────────────────────────
    // open:  open(/dev/bus/usb/BBB/DDD, O_RDWR)
    //        ioctl(USBDEVFS_DISCONNECT) to detach kernel driver (dvb_usb_rtl28xxu)
    //        ioctl(USBDEVFS_CLAIMINTERFACE, 0)

    [[nodiscard]] Result open(const UsbDeviceInfo& device);

    // close: ioctl(USBDEVFS_RELEASEINTERFACE, 0), ::close(fd)

    void close();

    // ── USB transfers ───────────────────────────────────────────────────────
    // control: ioctl(USBDEVFS_CONTROL, &usbdevfs_ctrltransfer{...})
    //   struct usbdevfs_ctrltransfer { u8 bRequestType, bRequest; u16 wValue, wIndex, wLength; u32 timeout; void* data; }

    [[nodiscard]] Result controlOut(std::uint16_t wValue, std::uint16_t wIndex, std::span<const std::uint8_t> data, std::uint16_t timeoutMs = 300);

    [[nodiscard]] std::expected<std::vector<std::uint8_t>, std::string> controlIn(std::uint16_t wValue, std::uint16_t wIndex, std::uint16_t length, std::uint16_t timeoutMs = 300);

    // bulk:   ioctl(USBDEVFS_BULK, &usbdevfs_bulktransfer{...})
    //   struct usbdevfs_bulktransfer { unsigned ep, len, timeout; void* data; }

    [[nodiscard]] std::expected<std::size_t, std::string> bulkRead(std::uint8_t endpoint, std::uint8_t* dst, std::size_t maxLen, std::uint32_t timeoutMs = 100);
};

} // namespace gr::blocks::rtlsdr

#endif // GNURADIO_LINUX_USB_DEVICE_HPP
