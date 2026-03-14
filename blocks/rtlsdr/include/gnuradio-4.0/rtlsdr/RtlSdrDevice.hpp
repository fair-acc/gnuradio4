#ifndef GNURADIO_RTLSDR_DEVICE_HPP
#define GNURADIO_RTLSDR_DEVICE_HPP

/**
 * @brief Hardware abstraction for the Realtek RTL2832U + Rafael Micro R820T/R828D USB dongle.
 *
 * Programs the RTL2832U demodulator and R820T/R828D tuner via USB->I2C passthrough of
 * the vendor control interface.
 * Register addresses and initialisation sequences are hardware facts documented in:
 *
 *   - RTL2832U Datasheet v1.4 (Realtek, 2010)
 *     https://homepages.uni-regensburg.de/~erc24492/SDR/Data_rtl2832u.pdf
 *   - R820T Datasheet (Rafael Micro, 2011)
 *     https://www.rtl-sdr.com/wp-content/uploads/2013/04/R820T_datasheet-Non_R-20111130_unlocked1.pdf
 *   - R820T2 Register Description (Rafael Micro, 2012)
 *     https://www.rtl-sdr.com/wp-content/uploads/2016/12/R820T2_Register_Description.pdf
 *
 * This is an independent Linux-only C++ reimplementation derived from
 * jtarrio/webrtlsdr (Apache-2.0, https://github.com/jtarrio/webrtlsdr),
 * which itself continues google/radioreceiver
 * (Apache-2.0, https://github.com/google/radioreceiver).
 *
 * No source code from osmocom/rtl-sdr or any GPL-licensed project has been
 * incorporated. See README.md for full provenance and acknowledgements.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <expected>
#include <print>
#include <span>
#include <string>
#include <thread>
#include <vector>

#if !defined(__EMSCRIPTEN__)
#include <libusb-1.0/libusb.h>
#else
#include <emscripten.h>
#include <emscripten/threading.h>
#endif

namespace gr::blocks::rtlsdr {

// ── RTL2832U hardware constants (datasheet section 10-11) ───────────────────

inline constexpr std::uint32_t kXtalFreq     = 28'800'000; // default crystal, Hz
inline constexpr std::uint32_t kIfFreq       = 3'570'000;  // R820T/R828D default IF, Hz
inline constexpr std::uint8_t  kBulkEndpoint = 0x81;
inline constexpr std::uint8_t  kWriteFlag    = 0x10;    // wIndex bit [4] for write ops
inline constexpr std::uint16_t kCtrlTimeout  = 300;     // ms
inline constexpr std::size_t   kBulkReadSize = 131'072; // 128 KB per transfer
inline constexpr std::size_t   kBulkInFlight = 12;

// USB control transfer block IDs (datasheet section 11, Table 28)
inline constexpr std::uint16_t kBlockUsb = 0x0100; // USB-side registers
inline constexpr std::uint16_t kBlockSys = 0x0200; // system / 8051 registers
inline constexpr std::uint16_t kBlockIic = 0x0600; // I2C master

// RTL2832U USB block registers (section 11, Tables 30-34)
inline constexpr std::uint16_t kUsbSysctl    = 0x2000; // USB system control
inline constexpr std::uint16_t kUsbEpaCtl    = 0x2148; // endpoint A control
inline constexpr std::uint16_t kUsbEpaMaxpkt = 0x2158; // endpoint A max packet size

// RTL2832U system registers (section 10, Table 24)
inline constexpr std::uint16_t kDemodCtl  = 0x3000; // PLL enable, ADC enable, hardware reset
inline constexpr std::uint16_t kDemodCtl1 = 0x300B; // IR wakeup, low-current crystal

// DEMOD_CTL bit field values (section 10.1, Table 24)
inline constexpr std::uint8_t kDemodCtlPowerOn  = 0xE8; // PLL_EN | ADC_I_EN | SOFT_RST_RELEASE | ADC_Q_EN
inline constexpr std::uint8_t kDemodCtlPowerOff = 0x20; // SOFT_RST_RELEASE only
inline constexpr std::uint8_t kDemodCtl1Init    = 0x22; // IR wakeup + low-current crystal

// EPA_CTL values
inline constexpr std::uint16_t kEpaCtlReset   = 0x1002; // stall + FIFO flush
inline constexpr std::uint16_t kEpaCtlRelease = 0x0000;
inline constexpr std::uint16_t kEpaMaxpkt512  = 0x0002; // 512-byte max packet

// I2C repeater gate (datasheet section 8.3, page 1, offset 0x01)
inline constexpr std::uint8_t kI2cRepeaterEnable  = 0x18; // IIC_repeat = 1
inline constexpr std::uint8_t kI2cRepeaterDisable = 0x10; // IIC_repeat = 0

// R820T/R828D I2C addresses and identification (R820T2 datasheet section 7.1)
inline constexpr std::uint8_t kR820tI2cAddr = 0x34;
inline constexpr std::uint8_t kR828dI2cAddr = 0x74;
inline constexpr std::uint8_t kTunerIdR820t = 0x69; // reg 0x00 after bit reversal

// R820T/R828D shadow register range
inline constexpr std::uint8_t kRegShadowStart = 0x05;
inline constexpr std::uint8_t kRegShadowEnd   = 0x1F;
inline constexpr std::size_t  kNumShadowRegs  = kRegShadowEnd - kRegShadowStart + 1; // 27

// PLL constants (R820T2 datasheet section 9)
inline constexpr std::uint32_t kVcoMin      = 1'770'000'000;
inline constexpr std::uint32_t kPllCalFreq  = 56'000'000;
inline constexpr std::uint8_t  kVcoPowerRef = 2; // R820T reference; R828D uses 1

inline constexpr std::array kKnownRtlSdrIds{
    std::pair<std::uint16_t, std::uint16_t>{0x0BDA, 0x2832},
    std::pair<std::uint16_t, std::uint16_t>{0x0BDA, 0x2838},
    std::pair<std::uint16_t, std::uint16_t>{0x0BDA, 0x2840},
};

// R820T/R828D initial shadow registers 0x05-0x1F (27 bytes, section 8)
inline constexpr std::array<std::uint8_t, kNumShadowRegs> kR8xxInitRegs{
    0x83, 0x32, 0x75,       // R5-R7:   LNA, power detector, mixer
    0xC0, 0x40, 0xD6, 0x6C, // R8-R11:  mixer buf, IF filter, channel filter, BW/HPF
    0xF5, 0x63, 0x75, 0x68, // R12-R15: VGA, LNA AGC, mixer AGC, clock control
    0x6C, 0x83, 0x80, 0x00, // R16-R19: PLL div, LDO_A, VCO/SDM, reserved
    0x0F, 0x00, 0xC0, 0x30, // R20-R23: integer div, SDM low, SDM high, LDO_D
    0x48, 0xCC, 0x60, 0x00, // R24-R27: reserved, RF filter, RF mux, tracking filter
    0x54, 0xAE, 0x4A, 0xC0, // R28-R31: PDET3, PDET1/2, PDET_CLK, LT attenuation
};

// RTL2832U FIR coefficients (symmetric 32-tap, packed into 20 bytes, section 4)
inline constexpr std::array<std::uint8_t, 20> kFirCoefficients{
    0xCA,
    0xDC,
    0xD7,
    0xD8,
    0xE0,
    0xF2,
    0x0E,
    0x35, // coefficients 0-7 (int8)
    0x06,
    0x50,
    0x9C,
    0x0D,
    0x71,
    0x11,
    0x14,
    0x71, // coefficients 8-15 (int12, packed)
    0x74,
    0x19,
    0x41,
    0xA5,
};

// nibble bit-reversal LUT for R820T register reads (section 7.3)
inline constexpr std::array<std::uint8_t, 16> kBitRevLut{
    0x0,
    0x8,
    0x4,
    0xC,
    0x2,
    0xA,
    0x6,
    0xE,
    0x1,
    0x9,
    0x5,
    0xD,
    0x3,
    0xB,
    0x7,
    0xF,
};

struct MuxConfig {
    std::uint32_t freqMhz;
    std::uint8_t  openDrain; // R23[3] OPEN_D
    std::uint8_t  rfMuxPoly; // R26    RFMUX + RFFILT
    std::uint8_t  tfC;       // R27    TF_NCH + TF_LP
};

// frequency-dependent RF frontend mux configuration (R820T characterisation, section 11)
inline constexpr std::array kMuxConfigs{
    MuxConfig{0, 0x08, 0x02, 0xDF},
    MuxConfig{50, 0x08, 0x02, 0xBE},
    MuxConfig{55, 0x08, 0x02, 0x8B},
    MuxConfig{60, 0x08, 0x02, 0x7B},
    MuxConfig{65, 0x08, 0x02, 0x69},
    MuxConfig{70, 0x08, 0x02, 0x58},
    MuxConfig{75, 0x00, 0x02, 0x44},
    MuxConfig{80, 0x00, 0x02, 0x44},
    MuxConfig{90, 0x00, 0x02, 0x34},
    MuxConfig{100, 0x00, 0x02, 0x34},
    MuxConfig{110, 0x00, 0x02, 0x24},
    MuxConfig{120, 0x00, 0x02, 0x24},
    MuxConfig{140, 0x00, 0x02, 0x14},
    MuxConfig{180, 0x00, 0x02, 0x13},
    MuxConfig{220, 0x00, 0x02, 0x13},
    MuxConfig{250, 0x00, 0x02, 0x11},
    MuxConfig{280, 0x00, 0x02, 0x00},
    MuxConfig{310, 0x00, 0x41, 0x00},
    MuxConfig{450, 0x00, 0x41, 0x00},
    MuxConfig{588, 0x00, 0x40, 0x00},
    MuxConfig{650, 0x00, 0x40, 0x00},
};

// ── WASM: inlined JS shims ──────────────────────────────────────────────────

#if defined(__EMSCRIPTEN__)

namespace detail {

struct SpscByteQueue {
    static constexpr std::size_t kCapacity = 4'194'304; // 4 MB
    static constexpr std::size_t kMask     = kCapacity - 1;
    static_assert((kCapacity & kMask) == 0);

    std::array<std::uint8_t, kCapacity> buf{};
    std::atomic<std::uint32_t>          head{0};
    std::atomic<std::uint32_t>          tail{0};

    std::size_t pop(std::uint8_t* dst, std::size_t maxLen) {
        const auto t   = tail.load(std::memory_order_relaxed);
        const auto h   = head.load(std::memory_order_acquire);
        const auto avl = static_cast<std::size_t>(h - t);
        const auto n   = std::min(maxLen, avl);
        if (n == 0) {
            return 0;
        }
        const auto start = static_cast<std::size_t>(t & kMask);
        const auto first = std::min(n, kCapacity - start);
        std::memcpy(dst, &buf[start], first);
        if (n > first) {
            std::memcpy(dst + first, buf.data(), n - first);
        }
        tail.store(t + static_cast<std::uint32_t>(n), std::memory_order_release);
        return n;
    }

    void reset() {
        head.store(0, std::memory_order_relaxed);
        tail.store(0, std::memory_order_relaxed);
    }
};

inline SpscByteQueue& iqQueue() {
    static SpscByteQueue q;
    return q;
}

// TODO: wasmDeviceReady global flag -- replace with proper device sharing
inline std::atomic<bool>& wasmDeviceReady() {
    static std::atomic<bool> ready{false};
    return ready;
}

} // namespace detail

// clang-format off

// thin WebUSB shims -- only raw USB primitives, no protocol logic

EM_ASYNC_JS(int, js_rtl_request_device, (), {
    if (!navigator || !navigator.usb) return -1;
    try {
        var filters = [
            { vendorId: 0x0BDA, productId: 0x2832 },
            { vendorId: 0x0BDA, productId: 0x2838 },
            { vendorId: 0x0BDA, productId: 0x2840 }
        ];
        Module._rtlDevice = await navigator.usb.requestDevice({ filters: filters });
        return 0;
    } catch (e) { return -1; }
});

EM_ASYNC_JS(int, js_rtl_get_device_count, (), {
    if (!navigator || !navigator.usb) return 0;
    try {
        var devs = await navigator.usb.getDevices();
        var pids = [0x2832, 0x2838, 0x2840];
        return devs.filter(function(d) { return d.vendorId === 0x0BDA && pids.indexOf(d.productId) >= 0; }).length;
    } catch (e) { return 0; }
});

EM_ASYNC_JS(int, js_rtl_open_device, (int idx), {
    try {
        if (Module._rtlDevice && Module._rtlDevice.opened) return 0;
        var devs = await navigator.usb.getDevices();
        var pids = [0x2832, 0x2838, 0x2840];
        var cands = devs.filter(function(d) { return d.vendorId === 0x0BDA && pids.indexOf(d.productId) >= 0; });
        if (idx >= cands.length) return -1;
        Module._rtlDevice = cands[idx];
        await Module._rtlDevice.open();
        if (Module._rtlDevice.configuration === null) await Module._rtlDevice.selectConfiguration(1);
        await Module._rtlDevice.claimInterface(0);
        return 0;
    } catch (e) { console.error("[RTL-SDR] open:", e.message); return -1; }
});

EM_ASYNC_JS(void, js_rtl_close_device, (), {
    if (!Module._rtlDevice) return;
    try { await Module._rtlDevice.releaseInterface(0); } catch (e) {}
    try { await Module._rtlDevice.close(); } catch (e) {}
    Module._rtlDevice = null;
});

EM_ASYNC_JS(int, js_rtl_ctrl_out, (int wValue, int wIndex, const uint8_t* dataPtr, int dataLen), {
    if (!Module._rtlDevice) return -1;
    var data = new Uint8Array(dataLen);
    for (var i = 0; i < dataLen; i++) data[i] = HEAPU8[dataPtr + i];
    var result = await Module._rtlDevice.controlTransferOut(
        { requestType: "vendor", recipient: "device", request: 0, value: wValue, index: wIndex }, data);
    return result.status === "ok" ? 0 : -1;
});

EM_ASYNC_JS(int, js_rtl_ctrl_in, (int wValue, int wIndex, int length, uint8_t* resultPtr), {
    if (!Module._rtlDevice) return -1;
    length = Math.max(length, 8);
    var result = await Module._rtlDevice.controlTransferIn(
        { requestType: "vendor", recipient: "device", request: 0, value: wValue, index: wIndex }, length);
    if (result.status !== "ok") return -1;
    var bytes = new Uint8Array(result.data.buffer);
    var n = Math.min(bytes.length, length);
    for (var i = 0; i < n; i++) HEAPU8[resultPtr + i] = bytes[i];
    return n;
});

EM_JS(int, js_rtl_get_product_name, (char* buf, int maxLen), {
    var name = Module._rtlDevice ? (Module._rtlDevice.productName || "RTL-SDR") : "";
    var bytes = (typeof TextEncoder !== "undefined") ? new TextEncoder().encode(name) : [];
    var n = Math.min(bytes.length, maxLen - 1);
    for (var i = 0; i < n; i++) HEAP8[buf + i] = bytes[i];
    HEAP8[buf + n] = 0;
    return n;
});

EM_JS(void, js_rtl_start_bulk_read, (int qBufOff, int headOff, int tailOff, int capacity, int mask, int readSize), {
    Module._rtlReading = true;
    var sab = Module.wasmMemory.buffer;
    var heapU8 = new Uint8Array(sab);
    var heap32 = new Uint32Array(sab);
    var headIdx = headOff >> 2;
    var tailIdx = tailOff >> 2;
    var totalBytes = 0;
    var totalXfers = 0;
    var droppedBytes = 0;
    var lastReport = performance.now();
    function submit() {
        if (!Module._rtlReading || !Module._rtlDevice) return;
        Module._rtlDevice.transferIn(1, readSize).then(function(r) {
            if (r.status === "ok" && r.data && r.data.byteLength > 0) {
                var data = new Uint8Array(r.data.buffer);
                var len = data.length;
                totalXfers++;
                if (heapU8.buffer !== Module.wasmMemory.buffer) {
                    sab = Module.wasmMemory.buffer;
                    heapU8 = new Uint8Array(sab);
                    heap32 = new Uint32Array(sab);
                }
                var h = Atomics.load(heap32, headIdx);
                var t = Atomics.load(heap32, tailIdx);
                var free = capacity - (h - t);
                var n = len < free ? len : free;
                if (n > 0) {
                    var s = h & mask;
                    var f = capacity - s;
                    if (f >= n) { heapU8.set(data.subarray(0, n), qBufOff + s); }
                    else { heapU8.set(data.subarray(0, f), qBufOff + s); heapU8.set(data.subarray(f, n), qBufOff); }
                    Atomics.store(heap32, headIdx, h + n);
                }
                totalBytes += n;
                if (n < len) droppedBytes += (len - n);
                var now = performance.now();
                if (now - lastReport > 5000) {
                    var el = (now - lastReport) / 1000;
                    console.log("[RTL-SDR] USB: " + (totalBytes / el / 1e6).toFixed(2) + " MB/s (" +
                        totalXfers + " xfers" + (droppedBytes > 0 ? ", dropped " + (droppedBytes/1024).toFixed(0) + " KB" : ", no drops") + ")");
                    totalBytes = 0; totalXfers = 0; droppedBytes = 0; lastReport = now;
                }
            }
            submit();
        }).catch(function(e) {
            if (Module._rtlReading) {
                Module._rtlDevice.clearHalt("in", 1).catch(function(){}).then(function() { if (Module._rtlReading) submit(); });
            }
        });
    }
    var nInFlight = 12;
    for (var k = 0; k < nInFlight; k++) submit();
});

EM_JS(void, js_rtl_stop_bulk_read, (), {
    Module._rtlReading = false;
});

// clang-format on

extern "C" {
EMSCRIPTEN_KEEPALIVE inline std::uint8_t*  rtlsdr_getQueueBuf() { return detail::iqQueue().buf.data(); }
EMSCRIPTEN_KEEPALIVE inline std::uint32_t* rtlsdr_getQueueHead() { return reinterpret_cast<std::uint32_t*>(&detail::iqQueue().head); }
EMSCRIPTEN_KEEPALIVE inline std::uint32_t* rtlsdr_getQueueTail() { return reinterpret_cast<std::uint32_t*>(&detail::iqQueue().tail); }
EMSCRIPTEN_KEEPALIVE inline int            rtlsdr_getQueueCapacity() { return static_cast<int>(detail::SpscByteQueue::kCapacity); }
EMSCRIPTEN_KEEPALIVE inline int            rtlsdr_getQueueMask() { return static_cast<int>(detail::SpscByteQueue::kMask); }
} // extern "C"

#endif // __EMSCRIPTEN__

// ── device abstraction ──────────────────────────────────────────────────────

enum class TunerType : std::uint8_t { none, r820t, r828d };

struct RtlSdrDevice {
    using Result      = std::expected<void, std::string>;
    using ValueResult = std::expected<double, std::string>;

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define GR_RTLSDR_TRY(expr)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    \
    do {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
        auto _r = (expr);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
        if (!_r)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
            return std::unexpected(_r.error());                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
    } while (0)
    // NOLINTEND(cppcoreguidelines-macro-usage)

#if !defined(__EMSCRIPTEN__)
    libusb_context*       _ctx    = nullptr;
    libusb_device_handle* _handle = nullptr;
#endif
    std::atomic<bool>                        _open{false};
    TunerType                                _tunerType    = TunerType::none;
    std::uint8_t                             _tunerI2cAddr = 0;
    std::array<std::uint8_t, kNumShadowRegs> _shadowRegs{};
    std::string                              _deviceName;

    RtlSdrDevice()                               = default;
    RtlSdrDevice(const RtlSdrDevice&)            = delete;
    RtlSdrDevice& operator=(const RtlSdrDevice&) = delete;
    ~RtlSdrDevice() { close(); }

    [[nodiscard]] bool isOpen() const { return _open.load(std::memory_order_acquire); }

    // ── lifecycle ───────────────────────────────────────────────────────────

    Result open(std::uint32_t deviceIndex = 0) {
#if !defined(__EMSCRIPTEN__)
        if (_handle) {
            return {};
        }
        if (!_ctx) {
            if (libusb_init(&_ctx) < 0) {
                return std::unexpected("libusb_init failed");
            }
        }
        // TODO: Linux-only libusb-free backend via /dev/bus/usb ioctl
        libusb_device** devList = nullptr;
        auto            nDevs   = libusb_get_device_list(_ctx, &devList);
        if (nDevs < 0) {
            return std::unexpected("libusb_get_device_list failed");
        }

        std::uint32_t  matchIdx = 0;
        libusb_device* target   = nullptr;
        for (ssize_t i = 0; i < nDevs; ++i) {
            libusb_device_descriptor desc{};
            if (libusb_get_device_descriptor(devList[i], &desc) < 0) {
                continue;
            }
            bool known = std::ranges::any_of(kKnownRtlSdrIds, [&](auto p) { return p.first == desc.idVendor && p.second == desc.idProduct; });
            if (known) {
                if (matchIdx == deviceIndex) {
                    target = devList[i];
                    break;
                }
                ++matchIdx;
            }
        }
        if (!target) {
            libusb_free_device_list(devList, 1);
            return std::unexpected("no RTL-SDR device found");
        }

        int ret = libusb_open(target, &_handle);
        libusb_free_device_list(devList, 1);
        if (ret < 0 || !_handle) {
            _handle = nullptr;
            return std::unexpected(std::format("libusb_open failed: {}", libusb_strerror(static_cast<libusb_error>(ret))));
        }

        if (libusb_kernel_driver_active(_handle, 0) == 1) {
            libusb_detach_kernel_driver(_handle, 0);
        }
        ret = libusb_claim_interface(_handle, 0);
        if (ret < 0) {
            libusb_close(_handle);
            _handle = nullptr;
            return std::unexpected(std::format("libusb_claim_interface failed: {}", libusb_strerror(static_cast<libusb_error>(ret))));
        }

        {
            libusb_device_descriptor devDesc{};
            libusb_get_device_descriptor(libusb_get_device(_handle), &devDesc);
            std::array<unsigned char, 128> nameBuf{};
            if (devDesc.iProduct > 0) {
                libusb_get_string_descriptor_ascii(_handle, devDesc.iProduct, nameBuf.data(), static_cast<int>(nameBuf.size()));
                _deviceName = reinterpret_cast<const char*>(nameBuf.data());
            }
        }
        if (_deviceName.empty()) {
            _deviceName = std::format("RTL-SDR #{}", deviceIndex);
        }

        auto initResult = initDevice();
        if (!initResult) {
            libusb_release_interface(_handle, 0);
            libusb_close(_handle);
            _handle = nullptr;
            return initResult;
        }
        _open.store(true, std::memory_order_release);
        std::println("[RTL-SDR] opened: {}", _deviceName);
        return {};
#else
        // TODO: WASM settings changes need main-thread proxy (Asyncify limitation)
        (void)deviceIndex;
        if (!detail::wasmDeviceReady().load(std::memory_order_acquire)) {
            return std::unexpected("device not yet authorized -- click Connect");
        }
        _open.store(true, std::memory_order_release);
        _deviceName = "RTL-SDR (WebUSB)";
        return {};
#endif
    }

    void close() {
#if !defined(__EMSCRIPTEN__)
        if (_handle) {
            libusb_release_interface(_handle, 0);
            libusb_close(_handle);
            _handle = nullptr;
        }
        if (_ctx) {
            libusb_exit(_ctx);
            _ctx = nullptr;
        }
#else
        if (_open.load()) {
            js_rtl_stop_bulk_read();
            js_rtl_close_device();
            detail::iqQueue().reset();
        }
#endif
        _open.store(false, std::memory_order_release);
        _tunerType    = TunerType::none;
        _tunerI2cAddr = 0;
    }

    // ── configuration ───────────────────────────────────────────────────────

    ValueResult setSampleRate(double rate) {
        auto   ratio    = static_cast<std::uint32_t>((static_cast<double>(kXtalFreq) * (1 << 22)) / rate) & 0x0FFFFFFCU;
        double realRate = (static_cast<double>(kXtalFreq) * (1 << 22)) / static_cast<double>(ratio);

        GR_RTLSDR_TRY(setDemodReg(1, 0x9F, ratio >> 16, 2));         // rsamp_ratio[27:16]
        GR_RTLSDR_TRY(setDemodReg(1, 0xA1, ratio & 0xFFFF, 2));      // rsamp_ratio[15:0]
        GR_RTLSDR_TRY(setDemodReg(1, 0x01, 0x14, 1));                // assert soft reset
        GR_RTLSDR_TRY(setDemodReg(1, 0x01, kI2cRepeaterDisable, 1)); // release soft reset
        std::println("[RTL-SDR] sample rate: {:.0f} Hz", realRate);
        return realRate;
    }

    ValueResult setCenterFrequency(double freq) {
        double tunerFreq = freq + kIfFreq;
        GR_RTLSDR_TRY(setMux(tunerFreq));
        auto pllResult = setPll(tunerFreq);
        if (!pllResult) {
            return std::unexpected(pllResult.error());
        }
        double actualFreq = *pllResult - kIfFreq;

        // pset_iffreq = -(IF * 2^22) / xtal (datasheet section 9.3)
        auto ifMultiplier = static_cast<std::int32_t>(-1.0 * static_cast<double>(kIfFreq) * (1 << 22) / kXtalFreq);
        ifMultiplier &= 0x3FFFFF;
        GR_RTLSDR_TRY(setDemodReg(1, 0x19, (ifMultiplier >> 16) & 0x3F, 1)); // pset_iffreq[21:16]
        GR_RTLSDR_TRY(setDemodReg(1, 0x1A, (ifMultiplier >> 8) & 0xFF, 1));  // pset_iffreq[15:8]
        GR_RTLSDR_TRY(setDemodReg(1, 0x1B, ifMultiplier & 0xFF, 1));         // pset_iffreq[7:0]
        std::println("[RTL-SDR] center freq: {:.0f} Hz", actualFreq);
        return actualFreq;
    }

    Result setGainMode(bool autoGain) {
        GR_RTLSDR_TRY(openI2C());
        if (autoGain) {
            GR_RTLSDR_TRY(writeTunerRegMask(0x05, 0x00, 0x10)); // R5: LNA_GAIN_MODE = auto
            GR_RTLSDR_TRY(writeTunerRegMask(0x07, 0x10, 0x10)); // R7: MIXGAIN_MODE = auto
            GR_RTLSDR_TRY(writeTunerRegMask(0x0C, 0x0B, 0x9F)); // R12: VGA_CODE = 26.5 dB
        }
        GR_RTLSDR_TRY(closeI2C());
        return {};
    }

    Result setTunerGain(float gainDb) {
        auto gainTenths = static_cast<int>(gainDb * 10.f);
        int  fullSteps  = std::clamp(gainTenths / 35, 0, 15);
        int  halfSteps  = (gainTenths - fullSteps * 35) >= 23 ? 1 : 0;
        auto lnaValue   = static_cast<std::uint8_t>(std::min(15, fullSteps + halfSteps));
        auto mixerValue = static_cast<std::uint8_t>(std::min(15, fullSteps));

        GR_RTLSDR_TRY(openI2C());
        GR_RTLSDR_TRY(writeTunerRegMask(0x05, 0x10, 0x10));       // R5: LNA_GAIN_MODE = manual
        GR_RTLSDR_TRY(writeTunerRegMask(0x05, lnaValue, 0x0F));   // R5: LNA_GAIN[3:0]
        GR_RTLSDR_TRY(writeTunerRegMask(0x07, 0x00, 0x10));       // R7: MIXGAIN_MODE = manual
        GR_RTLSDR_TRY(writeTunerRegMask(0x07, mixerValue, 0x0F)); // R7: MIX_GAIN[3:0]
        GR_RTLSDR_TRY(writeTunerRegMask(0x0C, 0x08, 0x9F));       // R12: VGA_CODE = 16.3 dB
        GR_RTLSDR_TRY(closeI2C());
        return {};
    }

    Result setAgcMode(bool on) {
        return setDemodReg(0, 0x19, on ? 0x25 : 0x05, 1); // sdr_ctrl: AGC enable/disable
    }

    Result setFreqCorrection(std::int32_t ppm) {
        auto offs = static_cast<std::int32_t>(ppm * -1.0 * (1 << 24) / 1'000'000.0);
        GR_RTLSDR_TRY(setDemodReg(1, 0x3F, offs & 0xFF, 1));        // samp_corr_l
        GR_RTLSDR_TRY(setDemodReg(1, 0x3E, (offs >> 8) & 0x3F, 1)); // samp_corr_h
        return {};
    }

    Result resetBuffer() {
        GR_RTLSDR_TRY(setUsbReg(kUsbEpaCtl, kEpaCtlReset, 2));
        GR_RTLSDR_TRY(setUsbReg(kUsbEpaCtl, kEpaCtlRelease, 2));
#if defined(__EMSCRIPTEN__)
        detail::iqQueue().reset();
#endif
        return {};
    }

    // ── data transfer ───────────────────────────────────────────────────────

    std::size_t readBulk(std::uint8_t* dst, std::size_t maxLen) {
#if !defined(__EMSCRIPTEN__)
        if (!_handle) {
            return 0;
        }
        int transferred = 0;
        int ret         = libusb_bulk_transfer(_handle, kBulkEndpoint, dst, static_cast<int>(maxLen), &transferred, 100);
        if (ret < 0 && ret != LIBUSB_ERROR_TIMEOUT) {
            return 0;
        }
        return transferred > 0 ? static_cast<std::size_t>(transferred) : 0;
#else
        return detail::iqQueue().pop(dst, maxLen);
#endif
    }

    void startBulkRead() {
#if defined(__EMSCRIPTEN__)
        auto& q = detail::iqQueue();
        q.reset();
        auto bufOff  = reinterpret_cast<std::uintptr_t>(q.buf.data());
        auto headOff = reinterpret_cast<std::uintptr_t>(&q.head);
        auto tailOff = reinterpret_cast<std::uintptr_t>(&q.tail);
        js_rtl_start_bulk_read(static_cast<int>(bufOff), static_cast<int>(headOff), static_cast<int>(tailOff), static_cast<int>(detail::SpscByteQueue::kCapacity), static_cast<int>(detail::SpscByteQueue::kMask), static_cast<int>(kBulkReadSize));
#endif
    }

    void stopBulkRead() {
#if defined(__EMSCRIPTEN__)
        js_rtl_stop_bulk_read();
#endif
    }

    // ── USB transport (the ONLY platform boundary) ──────────────────────────

    Result ctrlTransferOut(std::uint16_t wValue, std::uint16_t wIndex, std::span<const std::uint8_t> data) {
#if !defined(__EMSCRIPTEN__)
        int ret = libusb_control_transfer(_handle, static_cast<std::uint8_t>(LIBUSB_REQUEST_TYPE_VENDOR) | static_cast<std::uint8_t>(LIBUSB_ENDPOINT_OUT), 0, wValue, wIndex, const_cast<unsigned char*>(data.data()), static_cast<std::uint16_t>(data.size()), kCtrlTimeout);
        if (ret < 0) {
            return std::unexpected(std::format("ctrl out failed: {}", libusb_strerror(static_cast<libusb_error>(ret))));
        }
        return {};
#else
        int ret = js_rtl_ctrl_out(wValue, wIndex, data.data(), static_cast<int>(data.size()));
        if (ret < 0) {
            return std::unexpected("WebUSB ctrl out failed");
        }
        return {};
#endif
    }

    [[nodiscard]] std::expected<std::vector<std::uint8_t>, std::string> ctrlTransferIn(std::uint16_t wValue, std::uint16_t wIndex, std::uint16_t length) {
#if !defined(__EMSCRIPTEN__)
        std::vector<std::uint8_t> buf(std::max<std::uint16_t>(length, 8));
        int                       ret = libusb_control_transfer(_handle, static_cast<std::uint8_t>(LIBUSB_REQUEST_TYPE_VENDOR) | static_cast<std::uint8_t>(LIBUSB_ENDPOINT_IN), 0, wValue, wIndex, buf.data(), static_cast<std::uint16_t>(buf.size()), kCtrlTimeout);
        if (ret < 0) {
            return std::unexpected(std::format("ctrl in failed: {}", libusb_strerror(static_cast<libusb_error>(ret))));
        }
        buf.resize(static_cast<std::size_t>(ret));
        return buf;
#else
        std::vector<std::uint8_t> buf(static_cast<std::size_t>(std::max<int>(length, 8)));
        int                       ret = js_rtl_ctrl_in(wValue, wIndex, static_cast<int>(buf.size()), buf.data());
        if (ret < 0) {
            return std::unexpected("WebUSB ctrl in failed");
        }
        buf.resize(static_cast<std::size_t>(ret));
        return buf;
#endif
    }

    // ── RTL2832U register access (platform-agnostic) ────────────────────────

    Result setUsbReg(std::uint16_t addr, std::uint32_t value, std::uint8_t len) {
        std::array<std::uint8_t, 2> data{};
        if (len == 1) {
            data[0] = static_cast<std::uint8_t>(value & 0xFF);
        } else {
            data[0] = static_cast<std::uint8_t>(value >> 8);
            data[1] = static_cast<std::uint8_t>(value & 0xFF);
        }
        return ctrlTransferOut(addr, kBlockUsb | kWriteFlag, {data.data(), len});
    }

    Result setSysReg(std::uint16_t addr, std::uint8_t value) {
        std::array<std::uint8_t, 1> data{value};
        return ctrlTransferOut(addr, kBlockSys | kWriteFlag, data);
    }

    Result setDemodReg(std::uint8_t page, std::uint8_t addr, std::uint32_t value, std::uint8_t len) {
        std::array<std::uint8_t, 2> data{};
        if (len == 1) {
            data[0] = static_cast<std::uint8_t>(value & 0xFF);
        } else {
            data[0] = static_cast<std::uint8_t>(value >> 8);
            data[1] = static_cast<std::uint8_t>(value & 0xFF);
        }
        GR_RTLSDR_TRY(ctrlTransferOut(static_cast<std::uint16_t>((addr << 8) | 0x20), static_cast<std::uint16_t>(page | kWriteFlag), {data.data(), len}));
        return ctrlTransferIn(0x0120, 0x0A, 1).transform([](auto&&) {}); // read-back confirmation
    }

    // ── I2C access ──────────────────────────────────────────────────────────

    Result openI2C() { return setDemodReg(1, 0x01, kI2cRepeaterEnable, 1); }
    Result closeI2C() { return setDemodReg(1, 0x01, kI2cRepeaterDisable, 1); }

    Result setI2CReg(std::uint8_t i2cAddr, std::uint8_t reg, std::uint8_t value) {
        std::array<std::uint8_t, 2> data{reg, value};
        return ctrlTransferOut(i2cAddr, kBlockIic | kWriteFlag, data);
    }

    [[nodiscard]] std::expected<std::uint8_t, std::string> getI2CReg(std::uint8_t i2cAddr, std::uint8_t reg) {
        std::array<std::uint8_t, 1> regData{reg};
        GR_RTLSDR_TRY(ctrlTransferOut(i2cAddr, kBlockIic | kWriteFlag, regData));
        auto result = ctrlTransferIn(i2cAddr, kBlockIic, 1);
        if (!result) {
            return std::unexpected(result.error());
        }
        return result->empty() ? std::uint8_t{0} : (*result)[0];
    }

    [[nodiscard]] std::expected<std::vector<std::uint8_t>, std::string> getI2CRegBuf(std::uint8_t i2cAddr, std::uint8_t reg, std::uint8_t len) {
        std::array<std::uint8_t, 1> regData{reg};
        GR_RTLSDR_TRY(ctrlTransferOut(i2cAddr, kBlockIic | kWriteFlag, regData));
        return ctrlTransferIn(i2cAddr, kBlockIic, len);
    }

    // ── R820T tuner register access ─────────────────────────────────────────

    [[nodiscard]] static constexpr std::uint8_t bitRev(std::uint8_t b) { return static_cast<std::uint8_t>((kBitRevLut[b & 0xF] << 4) | kBitRevLut[b >> 4]); }

    [[nodiscard]] std::expected<std::vector<std::uint8_t>, std::string> readTunerBuf(std::uint8_t addr, std::uint8_t len) {
        auto raw = getI2CRegBuf(_tunerI2cAddr, addr, len);
        if (!raw) {
            return raw;
        }
        std::ranges::transform(*raw, raw->begin(), bitRev);
        return raw;
    }

    Result writeTunerRegMask(std::uint8_t addr, std::uint8_t value, std::uint8_t mask) {
        auto idx = static_cast<std::size_t>(addr - kRegShadowStart);
        if (idx >= _shadowRegs.size()) {
            return std::unexpected("tuner reg out of range");
        }
        _shadowRegs[idx] = (_shadowRegs[idx] & ~mask) | (value & mask);
        return setI2CReg(_tunerI2cAddr, addr, _shadowRegs[idx]);
    }

    Result writeTunerReg(std::uint8_t addr, std::uint8_t value) { return writeTunerRegMask(addr, value, 0xFF); }

    // ── init sequences ──────────────────────────────────────────────────────

    Result initDevice() {
        GR_RTLSDR_TRY(initDemod());
        GR_RTLSDR_TRY(detectTuner());
        if (_tunerType == TunerType::none) {
            return std::unexpected("no supported tuner found");
        }
        GR_RTLSDR_TRY(initTuner());
        return {};
    }

    Result initDemod() {
        // USB block init (datasheet section 11)
        GR_RTLSDR_TRY(setUsbReg(kUsbSysctl, 0x09, 1));
        GR_RTLSDR_TRY(setUsbReg(kUsbEpaMaxpkt, kEpaMaxpkt512, 2));
        GR_RTLSDR_TRY(setUsbReg(kUsbEpaCtl, kEpaCtlReset, 2));

        // power on demod (section 10)
        GR_RTLSDR_TRY(setSysReg(kDemodCtl1, kDemodCtl1Init));
        GR_RTLSDR_TRY(setSysReg(kDemodCtl, kDemodCtlPowerOn));

        // soft reset via I2C repeater register
        GR_RTLSDR_TRY(setDemodReg(1, 0x01, 0x14, 1));                // assert
        GR_RTLSDR_TRY(setDemodReg(1, 0x01, kI2cRepeaterDisable, 1)); // release

        // clear spectrum inversion, DDC shift, and IF registers
        GR_RTLSDR_TRY(setDemodReg(1, 0x15, 0x00, 1)); // spec_inv = normal
        for (std::uint8_t off = 0x16; off <= 0x1B; ++off) {
            GR_RTLSDR_TRY(setDemodReg(1, off, 0x00, 1));
        }

        // FIR coefficients (p1:0x1C-0x2F)
        for (std::size_t i = 0; i < kFirCoefficients.size(); ++i) {
            GR_RTLSDR_TRY(setDemodReg(1, static_cast<std::uint8_t>(0x1C + i), kFirCoefficients[i], 1));
        }

        // baseband and datapath configuration
        GR_RTLSDR_TRY(setDemodReg(0, 0x19, 0x05, 1)); // sdr_ctrl: SDR mode
        GR_RTLSDR_TRY(setDemodReg(1, 0x93, 0xF0, 1)); // fsm_state_0
        GR_RTLSDR_TRY(setDemodReg(1, 0x94, 0x0F, 1)); // fsm_state_1
        GR_RTLSDR_TRY(setDemodReg(1, 0x11, 0x00, 1)); // en_dagc = off
        GR_RTLSDR_TRY(setDemodReg(1, 0x04, 0x00, 1)); // agc_loop = off
        GR_RTLSDR_TRY(setDemodReg(0, 0x61, 0x60, 1)); // pid_filt = off
        GR_RTLSDR_TRY(setDemodReg(0, 0x06, 0x80, 1)); // opt_adc_iq = default
        GR_RTLSDR_TRY(setDemodReg(1, 0xB1, 0x1B, 1)); // en_bbin: zero-IF + DC cancel + IQ comp
        GR_RTLSDR_TRY(setDemodReg(0, 0x0D, 0x83, 1)); // clk_out: disable TP_CK0
        return {};
    }

    Result detectTuner() {
        _tunerType    = TunerType::none;
        _tunerI2cAddr = 0;

        struct Probe {
            std::uint8_t addr;
            std::uint8_t expected;
            TunerType    type;
        };
        constexpr std::array probes{
            Probe{kR820tI2cAddr, kTunerIdR820t, TunerType::r820t},
            Probe{kR828dI2cAddr, kTunerIdR820t, TunerType::r828d},
        };

        GR_RTLSDR_TRY(openI2C());
        for (const auto& [addr, expected, type] : probes) {
            auto id = getI2CReg(addr, 0x00);
            if (id && *id == expected) {
                _tunerI2cAddr = addr;
                _tunerType    = type;
                break;
            }
        }
        GR_RTLSDR_TRY(closeI2C());

        if (_tunerType != TunerType::none) {
            // R820T-specific demod settings
            GR_RTLSDR_TRY(setDemodReg(1, 0xB1, 0x1A, 1)); // en_bbin: disable zero-IF (use IF)
            GR_RTLSDR_TRY(setDemodReg(0, 0x08, 0x4D, 1)); // AD_EN_reg: I-ADC only
            GR_RTLSDR_TRY(setDemodReg(1, 0x15, 0x01, 1)); // spec_inv: enable spectrum inversion
            std::println("[RTL-SDR] found {} tuner", _tunerType == TunerType::r820t ? "R820T" : "R828D");
        }
        return {};
    }

    Result initTuner() {
        _shadowRegs = kR8xxInitRegs;

        GR_RTLSDR_TRY(openI2C());
        for (std::size_t i = 0; i < _shadowRegs.size(); ++i) {
            GR_RTLSDR_TRY(setI2CReg(_tunerI2cAddr, static_cast<std::uint8_t>(kRegShadowStart + i), _shadowRegs[i]));
        }

        // tuner electronics init (R820T2 register descriptions, sections 7.4-7.5)
        GR_RTLSDR_TRY(writeTunerRegMask(0x0C, 0x00, 0x0F)); // R12: clear xtal_check
        GR_RTLSDR_TRY(writeTunerRegMask(0x13, 0x03, 0x03)); // R19: VER_NUM
        GR_RTLSDR_TRY(writeTunerRegMask(0x1D, 0x00, 0x38)); // R29: PDET1_GAIN = lowest
        GR_RTLSDR_TRY(writeTunerRegMask(0x1C, 0x00, 0xF8)); // R28: PDET3_GAIN = lowest
        GR_RTLSDR_TRY(writeTunerRegMask(0x06, 0x10, 0x10)); // R6:  FILT_3DB = +3dB
        GR_RTLSDR_TRY(writeTunerRegMask(0x1A, 0x30, 0x30)); // R26: PLL_AUTO_CLK = 8kHz
        GR_RTLSDR_TRY(writeTunerRegMask(0x1D, 0xE5, 0xC7)); // R29: detect_bw + PDET1_GAIN + PDET2_GAIN
        GR_RTLSDR_TRY(writeTunerRegMask(0x1C, 0x24, 0xF8)); // R28: PDET3_GAIN adjust
        GR_RTLSDR_TRY(writeTunerRegMask(0x0D, 0x53, 0xFF)); // R13: LNA_VTHH + LNA_VTHL
        GR_RTLSDR_TRY(writeTunerRegMask(0x0E, 0x75, 0xFF)); // R14: MIX_VTH_H + MIX_VTH_L
        GR_RTLSDR_TRY(writeTunerRegMask(0x05, 0x00, 0x60)); // R5:  PWD_LNA1 = on, cable1 = off
        GR_RTLSDR_TRY(writeTunerRegMask(0x06, 0x00, 0x08)); // R6:  cable2_in = off
        GR_RTLSDR_TRY(writeTunerRegMask(0x11, 0x38, 0x38)); // R17: cp_cur = auto
        GR_RTLSDR_TRY(writeTunerRegMask(0x17, 0x30, 0x30)); // R23: div_buf_cur = 150uA
        GR_RTLSDR_TRY(writeTunerRegMask(0x0A, 0x40, 0x60)); // R10: PW_FILT = low current
        GR_RTLSDR_TRY(writeTunerRegMask(0x1E, 0x00, 0x60)); // R30: ext_enable = off
        GR_RTLSDR_TRY(closeI2C());

        // filter calibration at 56 MHz (section 10)
        auto calResult = setPll(kPllCalFreq);
        if (!calResult) {
            return std::unexpected(calResult.error());
        }

        GR_RTLSDR_TRY(openI2C());
        GR_RTLSDR_TRY(writeTunerRegMask(0x0B, 0x10, 0x10)); // R11: filt_cal_trig = 1
        GR_RTLSDR_TRY(writeTunerRegMask(0x0B, 0x00, 0x10)); // R11: filt_cal_trig = 0
        GR_RTLSDR_TRY(writeTunerRegMask(0x0F, 0x00, 0x08)); // R15: filt_cal_clk = off
        GR_RTLSDR_TRY(closeI2C());
        return {};
    }

    // ── PLL programming (R820T2 datasheet section 9) ────────────────────────

    ValueResult setPll(double freq) {
        std::uint32_t pllRef = kXtalFreq;
        int           divNum = 0;
        std::uint32_t mixDiv = 2;
        while (mixDiv <= 64) {
            if (freq * mixDiv >= static_cast<double>(kVcoMin)) {
                break;
            }
            mixDiv *= 2;
            ++divNum;
        }
        divNum = std::clamp(divNum, 0, 6);
        mixDiv = 1U << (divNum + 1);

        // VCO fine tune adjustment (section 9.2)
        GR_RTLSDR_TRY(openI2C());
        auto regBuf = readTunerBuf(0x00, 5);
        GR_RTLSDR_TRY(closeI2C());
        if (!regBuf) {
            return std::unexpected(regBuf.error());
        }
        int vcoFineTune = ((*regBuf)[4] & 0x30) >> 4;
        if (vcoFineTune > kVcoPowerRef) {
            --divNum;
        } else if (vcoFineTune < kVcoPowerRef) {
            ++divNum;
        }
        divNum = std::clamp(divNum, 0, 6);
        mixDiv = 1U << (divNum + 1);

        GR_RTLSDR_TRY(openI2C());
        GR_RTLSDR_TRY(writeTunerRegMask(0x10, static_cast<std::uint8_t>(divNum << 5), 0xE0)); // R16: SEL_DIV

        double vcoFreq = freq * mixDiv;
        auto   nint    = static_cast<std::uint32_t>(vcoFreq / (2.0 * pllRef));
        double vcoFra  = vcoFreq - 2.0 * pllRef * nint;
        if (nint < 13) {
            nint = 13;
        }

        auto ni = static_cast<std::uint8_t>((nint - 13) / 4);
        auto si = static_cast<std::uint8_t>((nint - 13) % 4);
        GR_RTLSDR_TRY(writeTunerReg(0x14, static_cast<std::uint8_t>(ni | (si << 6)))); // R20: NI2C + SI2C

        // sigma-delta modulator (section 9.3)
        auto sdm = static_cast<std::uint16_t>(std::min(65535.0, 32768.0 * vcoFra / pllRef));
        GR_RTLSDR_TRY(writeTunerRegMask(0x12, sdm == 0 ? 0x08 : 0x00, 0x08));             // R18: PW_SDM
        GR_RTLSDR_TRY(writeTunerReg(0x16, static_cast<std::uint8_t>((sdm >> 8) & 0xFF))); // R22: SDM_IN[15:8]
        GR_RTLSDR_TRY(writeTunerReg(0x15, static_cast<std::uint8_t>(sdm & 0xFF)));        // R21: SDM_IN[7:0]

        // PLL lock check (section 9.5)
        regBuf = readTunerBuf(0x00, 3);
        if (regBuf && !((*regBuf)[2] & 0x40)) {
            GR_RTLSDR_TRY(writeTunerRegMask(0x12, 0x60, 0xE0)); // R18: increase VCO current
        }
        GR_RTLSDR_TRY(closeI2C());

        double actualFreq = (2.0 * pllRef * (nint + static_cast<double>(sdm) / 65536.0)) / mixDiv;
        return actualFreq;
    }

    // ── mux configuration (section 11) ──────────────────────────────────────

    Result setMux(double freq) {
        auto        freqMhz = static_cast<std::uint32_t>(freq / 1e6);
        const auto* cfg     = &kMuxConfigs[0];
        for (const auto& entry : kMuxConfigs) {
            if (entry.freqMhz <= freqMhz) {
                cfg = &entry;
            } else {
                break;
            }
        }

        GR_RTLSDR_TRY(openI2C());
        GR_RTLSDR_TRY(writeTunerRegMask(0x17, cfg->openDrain, 0x08)); // R23: OPEN_D
        GR_RTLSDR_TRY(writeTunerRegMask(0x1A, cfg->rfMuxPoly, 0xC3)); // R26: RFMUX + RFFILT
        GR_RTLSDR_TRY(writeTunerRegMask(0x1B, cfg->tfC, 0xFF));       // R27: TF_NCH + TF_LP
        GR_RTLSDR_TRY(writeTunerRegMask(0x10, 0x00, 0x0B));           // R16: CAPX = 0pF
        GR_RTLSDR_TRY(writeTunerRegMask(0x08, 0x00, 0x3F));           // R8:  IMR_G = 0
        GR_RTLSDR_TRY(writeTunerRegMask(0x09, 0x00, 0x3F));           // R9:  IMR_P = 0
        GR_RTLSDR_TRY(closeI2C());
        return {};
    }

#undef GR_RTLSDR_TRY
};

} // namespace gr::blocks::rtlsdr

#endif // GNURADIO_RTLSDR_DEVICE_HPP
