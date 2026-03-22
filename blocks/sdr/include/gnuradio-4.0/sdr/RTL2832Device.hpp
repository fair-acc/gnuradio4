#ifndef GNURADIO_RTL2832_DEVICE_HPP
#define GNURADIO_RTL2832_DEVICE_HPP

/**
 * @brief Hardware abstraction for the Realtek RTL2832U and R820T/R828D/R860 and E4000 tuner SDR dongles.
 *
 * Programs the RTL2832U demodulator and attached tuner via USB->I2C passthrough of vendor control transfers.
 * Native: Linux USB ioctl (zero-dependency). WASM: WebUSB via thin JS shims.
 *
 * This implementation is based on information from and TypeScript to C++23 translation of:
 * https://github.com/jtarrio/webrtlsdr
 *
 * Datasheet references:
 *   - RTL2832U Datasheet v1.4 (Realtek, 2010)
 *     https://homepages.uni-regensburg.de/~erc24492/SDR/Data_rtl2832u.pdf
 *   - R820T Datasheet (Rafael Micro, 2011)
 *     https://www.rtl-sdr.com/wp-content/uploads/2013/04/R820T_datasheet-Non_R-20111130_unlocked1.pdf
 *   - R820T2 Register Description (Rafael Micro, 2012)
 *     https://www.rtl-sdr.com/wp-content/uploads/2016/12/R820T2_Register_Description.pdf
 *   - Elonics E4000 — Low-Power CMOS Multi-Band Tuner, DS-E4000-DS001, v4.0, Aug 2010
 *     https://www.nooelec.com/files/e4000datasheet.pdf
 *
 * N.B. the FC0012/FC0013/FC2580 family of tuners are not supported (more complex HW, quirks, no datasheet).
 *
 * See Readme.md for full provenance and acknowledgements.
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

#include <gnuradio-4.0/common/USBDevice.hpp>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#include <emscripten/threading.h>
#endif

namespace gr::blocks::sdr {

// RTL2832U demodulator constants
// ref: RTL2832U Datasheet v1.4 (Realtek, 2010), https://homepages.uni-regensburg.de/~erc24492/SDR/Data_rtl2832u.pdf

inline constexpr std::uint32_t kXtalFreq     = 28'800'000; // default crystal, Hz
inline constexpr std::uint32_t kIfFreq       = 3'570'000;  // R820T/R828D default IF, Hz
inline constexpr std::uint8_t  kBulkEndpoint = 0x81;
inline constexpr std::uint8_t  kWriteFlag    = 0x10;    // wIndex bit[4] for write ops (Table 28, p. 43)
inline constexpr std::uint16_t kCtrlTimeout  = 300;     // ms
inline constexpr std::uint8_t  kVendorOut    = 0x40;    // USB vendor request, host→device
inline constexpr std::uint8_t  kVendorIn     = 0xC0;    // USB vendor request, device→host
inline constexpr std::size_t   kBulkReadSize = 131'072; // 128 KB per transfer

// USB vendor command block IDs (Table 27, p. 42; wIndex encoding in Table 28, p. 43)
inline constexpr std::uint16_t kBlockUsb = 0x0100; // USB-side registers
inline constexpr std::uint16_t kBlockSys = 0x0200; // system / 8051 registers
inline constexpr std::uint16_t kBlockIic = 0x0600; // I2C master

// USB SIE registers (Table 29, p. 44; bit fields in Tables 30-33, p. 45-46)
inline constexpr std::uint16_t kUsbSysctl    = 0x2000; // USB system control
inline constexpr std::uint16_t kUsbEpaCtl    = 0x2148; // endpoint A control
inline constexpr std::uint16_t kUsbEpaMaxpkt = 0x2158; // endpoint A max packet size

// system registers (Table 14, p. 33; bit fields in Table 15, p. 35)
inline constexpr std::uint16_t kDemodCtl  = 0x3000; // PLL enable, ADC enable, hardware reset
inline constexpr std::uint16_t kDemodCtl1 = 0x300B; // IR wakeup, low-current crystal

// DEMOD_CTL bit fields (Table 15, p. 35)
inline constexpr std::uint8_t kDemodCtlPowerOn = 0xE8; // PLL_EN | ADC_I_EN | SOFT_RST_RELEASE | ADC_Q_EN
inline constexpr std::uint8_t kDemodCtl1Init   = 0x22; // IR wakeup + low-current crystal

// EPA_CTL bit fields (Table 32, p. 46)
inline constexpr std::uint16_t kEpaCtlReset   = 0x1002; // stall + FIFO flush
inline constexpr std::uint16_t kEpaCtlRelease = 0x0000;
inline constexpr std::uint16_t kEpaMaxpkt512  = 0x0002; // 512-byte max packet (Table 33, p. 46)

// I2C repeater gate (Table 3, p. 21; demod page 1, offset 0x01, bit[3])
inline constexpr std::uint8_t kI2cRepeaterEnable  = 0x18; // IIC_repeat = 1
inline constexpr std::uint8_t kI2cRepeaterDisable = 0x10; // IIC_repeat = 0

// R820T/R828D tuner constants
// ref: R820T2 Register Description (Rafael Micro, 2012),
//      https://www.rtl-sdr.com/wp-content/uploads/2016/12/R820T2_Register_Description.pdf

inline constexpr std::uint8_t kR820tI2cAddr = 0x34; // 8-bit write address (Table 1-1, p. 2)
inline constexpr std::uint8_t kR828dI2cAddr = 0x74; // R828D dual-tuner I2C address
inline constexpr std::uint8_t kTunerIdR820t = 0x69; // reg R0 value 0x96 bit-reversed (p. 4-5)

inline constexpr std::uint8_t kRegShadowStart = 0x05; // writable register range (Table 1-2, p. 5)
inline constexpr std::uint8_t kRegShadowEnd   = 0x1F;
inline constexpr std::size_t  kNumShadowRegs  = kRegShadowEnd - kRegShadowStart + 1; // 27

// PLL constants (Table 1-3, R16/R20-R22, p. 9)
inline constexpr std::uint32_t kVcoMin      = 1'770'000'000;
inline constexpr std::uint32_t kPllCalFreq  = 56'000'000;
inline constexpr std::uint8_t  kVcoPowerRef = 2; // R820T reference; R828D uses 1

inline constexpr std::array kKnownRTL2832Ids{
    common::USBDeviceId{0x0BDA, 0x2832, "RTL2832U"},
    common::USBDeviceId{0x0BDA, 0x2838, "RTL2838UHIDIR"},
    common::USBDeviceId{0x0BDA, 0x2840, "RTL2840"},
};

// clang-format off
// R820T/R828D initial shadow registers 0x05-0x1F (27 bytes; operating configuration for SDR use)
inline constexpr std::array<std::uint8_t, kNumShadowRegs> kR8xxInitRegs{
    0x83, 0x32, 0x75, // R5-R7:   LNA, power detector, mixer
    0xC0, 0x40, 0xD6, 0x6C, // R8-R11:  mixer buf, IF filter, channel filter, BW/HPF
    0xF5, 0x63, 0x75, 0x68, // R12-R15: VGA, LNA AGC, mixer AGC, clock control
    0x6C, 0x83, 0x80, 0x00, // R16-R19: PLL div, LDO_A, VCO/SDM, reserved
    0x0F, 0x00, 0xC0, 0x30, // R20-R23: integer div, SDM low, SDM high, LDO_D
    0x48, 0xCC, 0x60, 0x00, // R24-R27: reserved, RF filter, RF mux, tracking filter
    0x54, 0xAE, 0x4A, 0xC0, // R28-R31: PDET3, PDET1/2, PDET_CLK, LT attenuation
};

// RTL2832U LPF coefficients (symmetric 32-tap, packed into 20 bytes, demod page 1, 0x1C-0x2F)
inline constexpr std::array<std::uint8_t, 20> kFirCoefficients{
    0xCA, 0xDC, 0xD7, 0xD8, 0xE0, 0xF2, 0x0E, 0x35, // coefficients 0-7 (int8)
    0x06, 0x50, 0x9C, 0x0D, 0x71, 0x11, 0x14, 0x71, // coefficients 8-15 (int12, packed)
    0x74, 0x19, 0x41, 0xA5,
};

// nibble bit-reversal LUT for R820T register reads (LSB-first transmission, p. 4)
inline constexpr std::array<std::uint8_t, 16> kBitRevLut{ 0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE, 0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF,};

struct MuxConfig {
    std::uint32_t freqMhz;
    std::uint8_t  openDrain; // R23[3] OPEN_D
    std::uint8_t  rfMuxPoly; // R26    RFMUX + RFFILT
    std::uint8_t  tfC;       // R27    TF_NCH + TF_LP
};

// frequency-dependent RF frontend mux configuration (R26/R27 fields, Table 1-3, p. 10)
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

// E4000 tuner constants
// ref: Elonics E4000 datasheet v4.0 (DS-E4000-DS001), https://www.nooelec.com/files/e4000datasheet.pdf

inline constexpr std::uint8_t kE4kI2cAddr = 0xC8; // 8-bit write address, 7-bit: 0x64 (Table 1, p. 23)
inline constexpr std::uint8_t kE4kChipId  = 0x40; // register 0x02 'Master3' (register map, p. 17)

// synthesizer registers (register map, p. 17-18; sections 1.3-1.7, p. 25-26)
inline constexpr std::uint8_t kE4kRegSynth1 = 0x07; // PLL lock [0], band [2:1]
inline constexpr std::uint8_t kE4kRegSynth3 = 0x09; // integer divider Z
inline constexpr std::uint8_t kE4kRegSynth4 = 0x0A; // fractional X [7:0]
inline constexpr std::uint8_t kE4kRegSynth5 = 0x0B; // fractional X [15:8]
inline constexpr std::uint8_t kE4kRegSynth7 = 0x0D; // VCO divider select + 3-phase

// filter registers (register map, p. 18; Tables 21-24, p. 43-46)
inline constexpr std::uint8_t kE4kRegFilt1 = 0x10; // RF filter [3:0]
inline constexpr std::uint8_t kE4kRegFilt2 = 0x11; // mixer BW [7:4], RC BW [3:0]
inline constexpr std::uint8_t kE4kRegFilt3 = 0x12; // channel BW [4:0]

// gain registers (register map, p. 18; Table 6, p. 31; section 1.15-1.17, p. 34-35)
inline constexpr std::uint8_t kE4kRegGain1 = 0x14; // LNA [3:0]
inline constexpr std::uint8_t kE4kRegGain2 = 0x15; // mixer [0]: 0=4dB, 1=12dB
inline constexpr std::uint8_t kE4kRegGain3 = 0x16; // IF stages 1–4
inline constexpr std::uint8_t kE4kRegGain4 = 0x17; // IF stages 5–6

// AGC registers (register map, p. 19; Table 5, p. 30; section 1.15.2, p. 35)
inline constexpr std::uint8_t kE4kRegAgc1 = 0x1A; // AGC mode [3:0]
inline constexpr std::uint8_t kE4kRegAgc7 = 0x20; // mixer gain auto [0]

// DC offset calibration (register map, p. 19-20; sections 1.25-1.27, p. 48-49)
inline constexpr std::uint8_t kE4kRegDc1 = 0x29; // cal trigger [0]
inline constexpr std::uint8_t kE4kRegDc5 = 0x2D; // LUT enable flags

// miscellaneous (register map, p. 21; section 1.28, p. 52)
inline constexpr std::uint8_t kE4kRegBias       = 0x78;
inline constexpr std::uint8_t kE4kRegClkoutPwdn = 0x7A;

// E4000 frequency bands
enum class E4kBand : std::uint8_t { vhf2 = 0, vhf3 = 1, uhf = 2, lBand = 3 };

// PLL VCO output divider lookup table (Table 2, p. 26)
// the E4000 VCO output is divided by 'mul' to produce f_LO
struct E4kPllEntry {
    std::uint32_t maxFreqKhz; // upper bound of this entry
    std::uint8_t  synth7;     // SYNTH7 register value (VCO divider + 3-phase control)
    std::uint32_t mul;        // f_VCO / f_LO ratio
};

inline constexpr std::array kE4kPllLut{
    E4kPllEntry{72'400, 0x0F, 48},
    E4kPllEntry{81'200, 0x0E, 40},
    E4kPllEntry{108'300, 0x0D, 32},
    E4kPllEntry{162'500, 0x0C, 24},
    E4kPllEntry{216'600, 0x0B, 16},
    E4kPllEntry{325'000, 0x0A, 12},
    E4kPllEntry{350'000, 0x09, 8},
    E4kPllEntry{432'000, 0x03, 8},
    E4kPllEntry{667'000, 0x02, 6},
    E4kPllEntry{1'200'000, 0x01, 4},
};

// RF filter centre frequencies per band (Table 21, p. 43)
// UHF and L-band use nearest-match selection; VHF uses a fixed filter (index 0)
inline constexpr std::array<std::uint32_t, 16> kE4kRfFilterUhfMhz{
    360, 380, 405, 425, 450, 475, 505, 540, 575, 615, 670, 720, 760, 840, 890, 970,
};
inline constexpr std::array<std::uint32_t, 16> kE4kRfFilterLbandMhz{
    1300, 1320, 1360, 1410, 1445, 1460, 1490, 1530, 1560, 1590, 1640, 1660, 1680, 1700, 1720, 1750,
};

// LNA gain per GAIN1 register index (tenths of dB, Table 6, p. 31)
// indices 2–3 are undocumented intermediate steps
inline constexpr std::array<std::int16_t, 15> kE4kLnaGainTenths{
    -50, -25, 0, 0, 0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300,
};
// clang-format on

// WASM: inlined JS shims

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

// signals that the main-thread button handler has opened the WebUSB device (user gesture required)
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
    } catch (e) { console.error("[RTL2832] open:", e.message); return -1; }
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
    var name = Module._rtlDevice ? (Module._rtlDevice.productName || "RTL2832") : "";
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
                    console.log("[RTL2832] USB: " + (totalBytes / el / 1e6).toFixed(2) + " MB/s (" +
                        totalXfers + " xfers" + (droppedBytes > 0 ? ", dropped " + (droppedBytes/1024).toFixed(0) + " KB" : ", no drops") + ")");
                    totalBytes = 0; totalXfers = 0; droppedBytes = 0; lastReport = now;
                }
            }
            submit();
        }).catch(function(e) {
            if (!Module._rtlReading) return;
            if (e.name === "NotFoundError" || e.name === "NetworkError") {
                console.error("[RTL2832] device disconnected:", e.message);
                Module._rtlReading = false;
                return;
            }
            Module._rtlDevice.clearHalt("in", 1).catch(function(){}).then(function() { if (Module._rtlReading) submit(); });
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
EMSCRIPTEN_KEEPALIVE inline std::uint8_t*  rtl2832_getQueueBuf() { return detail::iqQueue().buf.data(); }
EMSCRIPTEN_KEEPALIVE inline std::uint32_t* rtl2832_getQueueHead() { return reinterpret_cast<std::uint32_t*>(&detail::iqQueue().head); }
EMSCRIPTEN_KEEPALIVE inline std::uint32_t* rtl2832_getQueueTail() { return reinterpret_cast<std::uint32_t*>(&detail::iqQueue().tail); }
EMSCRIPTEN_KEEPALIVE inline int            rtl2832_getQueueCapacity() { return static_cast<int>(detail::SpscByteQueue::kCapacity); }
EMSCRIPTEN_KEEPALIVE inline int            rtl2832_getQueueMask() { return static_cast<int>(detail::SpscByteQueue::kMask); }
} // extern "C"

#endif // __EMSCRIPTEN__

// device abstraction

enum class TunerType : std::uint8_t {
    none,
    r820t, // R820T/R820T2/R860 single-tuner family (register-compatible)
    r828d, // R828D dual-tuner variant (I2C addr 0x74, VCO power ref 1)
    e4000  // Elonics E4000 zero-IF tuner
};

struct RTL2832Device {
    using Result      = std::expected<void, std::string>;
    using ValueResult = std::expected<double, std::string>;

    struct DemodWrite {
        std::uint8_t  page;
        std::uint8_t  addr;
        std::uint32_t value;
        std::uint8_t  len = 1;
    };
    struct TunerWrite {
        std::uint8_t addr;
        std::uint8_t value;
        std::uint8_t mask = 0xFF;
    };
    struct E4kWrite {
        std::uint8_t reg;
        std::uint8_t value;
    };

#if !defined(__EMSCRIPTEN__)
    common::USBDevice _usb;
#endif
    std::atomic<bool>                        _open{false};
    TunerType                                _tunerType    = TunerType::none;
    std::uint8_t                             _tunerI2cAddr = 0;
    std::array<std::uint8_t, kNumShadowRegs> _shadowRegs{};
    std::string                              _deviceName;

    RTL2832Device()                                = default;
    RTL2832Device(const RTL2832Device&)            = delete;
    RTL2832Device& operator=(const RTL2832Device&) = delete;
    ~RTL2832Device() { close(); }

    [[nodiscard]] bool isOpen() const { return _open.load(std::memory_order_acquire); }

    // lifecycle

    Result open([[maybe_unused]] std::uint32_t deviceIndex = 0) {
#if !defined(__EMSCRIPTEN__)
        if (_usb.isOpen()) {
            return {};
        }
        auto devices = common::enumerateUSBDevices(kKnownRTL2832Ids);
        if (deviceIndex >= devices.size()) {
            return std::unexpected("no RTL2832 device found");
        }
        auto& info = devices[deviceIndex];
        if (auto r = _usb.open(info); !r) {
            return std::unexpected(r.error());
        }
        _deviceName = info.product.empty() ? std::format("RTL2832 #{}", deviceIndex) : info.product;

        if (auto r = initDevice(); !r) {
            _usb.close();
            return r;
        }
        _open.store(true, std::memory_order_release);
        std::println("[RTL2832] opened: {}", _deviceName);
        return {};
#else
        if (!detail::wasmDeviceReady().load(std::memory_order_acquire)) {
            return std::unexpected("device not yet authorized -- click Connect");
        }
        _open.store(true, std::memory_order_release);
        char nameBuf[128]{};
        js_rtl_get_product_name(nameBuf, sizeof(nameBuf));
        _deviceName = std::strlen(nameBuf) > 0 ? std::string(nameBuf) : "RTL2832 (WebUSB)";
        // detect tuner so settings changes (gain, frequency) work from the IO thread
        if (auto r = initDevice(); !r) {
            std::println(stderr, "[RTL2832] WASM initDevice: {}", r.error());
            // non-fatal: basic demod settings still work without tuner detection
        }
        return {};
#endif
    }

    void close() {
#if !defined(__EMSCRIPTEN__)
        _usb.close();
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

    // configuration

    ValueResult setSampleRate(float rate) {
        auto   ratio    = static_cast<std::uint32_t>((static_cast<double>(kXtalFreq) * (1 << 22)) / static_cast<double>(rate)) & 0x0FFFFFFCU;
        double realRate = (static_cast<double>(kXtalFreq) * (1 << 22)) / static_cast<double>(ratio);

        if (auto r = writeDemodBatch({
                {1, 0x9F, ratio >> 16, 2},      // rsamp_ratio[27:16]
                {1, 0xA1, ratio & 0xFFFF, 2},   // rsamp_ratio[15:0]
                {1, 0x01, 0x14},                // assert soft reset
                {1, 0x01, kI2cRepeaterDisable}, // release soft reset
            });
            !r) {
            return std::unexpected(r.error());
        }
        return realRate;
    }

    ValueResult setCenterFrequency(double freq) {
        if (_tunerType == TunerType::e4000) {
            // E4000: zero-IF tuner — no IF offset, direct LO frequency
            auto pllResult = setE4kPll(freq);
            if (!pllResult) {
                return std::unexpected(pllResult.error());
            }
            if (auto r = setE4kBandFilter(freq); !r) {
                return std::unexpected(r.error());
            }
            if (auto r = writeDemodBatch({{1, 0x19, 0x00}, {1, 0x1A, 0x00}, {1, 0x1B, 0x00}}); !r) { // pset_iffreq = 0 (zero-IF)
                return std::unexpected(r.error());
            }
            return *pllResult;
        }

        // R820T/R828D: low-IF tuner — add 3.57 MHz IF offset
        double tunerFreq = freq + kIfFreq;
        if (auto r = setMux(tunerFreq); !r) {
            return std::unexpected(r.error());
        }
        auto pllResult = setPll(tunerFreq);
        if (!pllResult) {
            return std::unexpected(pllResult.error());
        }
        double actualFreq = *pllResult - kIfFreq;

        // pset_iffreq = -(IF * 2^22) / xtal (datasheet section 9.3)
        auto ifMul = static_cast<std::int32_t>(-1.0 * static_cast<double>(kIfFreq) * (1 << 22) / kXtalFreq) & 0x3FFFFF;
        if (auto r = writeDemodBatch({
                {1, 0x19, static_cast<std::uint32_t>((ifMul >> 16) & 0x3F)}, // pset_iffreq[21:16]
                {1, 0x1A, static_cast<std::uint32_t>((ifMul >> 8) & 0xFF)},  // pset_iffreq[15:8]
                {1, 0x1B, static_cast<std::uint32_t>(ifMul & 0xFF)},         // pset_iffreq[7:0]
            });
            !r) {
            return std::unexpected(r.error());
        }
        return actualFreq;
    }

    Result setGainMode(bool autoGain) {
        auto gate = i2cGate();
        if (!gate) {
            return std::unexpected(gate.error());
        }

        if (_tunerType == TunerType::e4000) {
            return autoGain ? writeE4kBatch({{kE4kRegAgc1, 0x09}, {kE4kRegAgc7, 0x01}})  // auto LNA + auto mixer
                            : writeE4kBatch({{kE4kRegAgc1, 0x00}, {kE4kRegAgc7, 0x00}}); // full manual
        }

        // R820T/R828D
        if (autoGain) {
            return writeTunerBatch({
                {0x05, 0x00, 0x10}, // R5: LNA_GAIN_MODE = auto
                {0x07, 0x10, 0x10}, // R7: MIXGAIN_MODE = auto
                {0x0C, 0x0B, 0x9F}, // R12: VGA_CODE = 26.5 dB
            });
        }
        return {};
    }

    Result setTunerGain(float gainDb) {
        if (_tunerType == TunerType::e4000) {
            // find closest LNA gain step
            auto         gainTenths = static_cast<int>(gainDb * 10.f);
            std::uint8_t lnaIdx     = 0;
            int          bestDiff   = std::abs(gainTenths - static_cast<int>(kE4kLnaGainTenths[0]));
            for (std::size_t i = 1; i < kE4kLnaGainTenths.size(); ++i) {
                int diff = std::abs(gainTenths - static_cast<int>(kE4kLnaGainTenths[i]));
                if (diff < bestDiff) {
                    bestDiff = diff;
                    lnaIdx   = static_cast<std::uint8_t>(i);
                }
            }
            std::uint8_t mixerReg = gainDb > 20.f ? 0x01 : 0x00; // 12 dB or 4 dB

            auto gate = i2cGate();
            if (!gate) {
                return std::unexpected(gate.error());
            }
            return writeE4kBatch({
                {kE4kRegGain1, lnaIdx}, {kE4kRegGain2, mixerReg}, {kE4kRegGain3, 0x01}, // IF stage1=6dB, stages2-4=0dB
                {kE4kRegGain4, 0x12},                                                   // IF stages5-6=9dB each
            });
        }

        // R820T/R828D
        auto gainTenths = static_cast<int>(gainDb * 10.f);
        int  fullSteps  = std::clamp(gainTenths / 35, 0, 15);
        int  halfSteps  = (gainTenths - fullSteps * 35) >= 23 ? 1 : 0;
        auto lnaValue   = static_cast<std::uint8_t>(std::min(15, fullSteps + halfSteps));
        auto mixerValue = static_cast<std::uint8_t>(std::min(15, fullSteps));

        auto gate = i2cGate();
        if (!gate) {
            return std::unexpected(gate.error());
        }
        return writeTunerBatch({
            {0x05, 0x10, 0x10},       // R5: LNA_GAIN_MODE = manual
            {0x05, lnaValue, 0x0F},   // R5: LNA_GAIN[3:0]
            {0x07, 0x00, 0x10},       // R7: MIXGAIN_MODE = manual
            {0x07, mixerValue, 0x0F}, // R7: MIX_GAIN[3:0]
            {0x0C, 0x08, 0x9F},       // R12: VGA_CODE = 16.3 dB
        });
    }

    Result setAgcMode(bool on) {
        return setDemodReg(0, 0x19, on ? 0x25 : 0x05, 1); // sdr_ctrl: AGC enable/disable
    }

    Result setFreqCorrection(std::int32_t ppm) {
        auto offs = static_cast<std::int32_t>(ppm * -1.0 * (1 << 24) / 1'000'000.0);
        return writeDemodBatch({
            {1, 0x3F, static_cast<std::uint32_t>(offs & 0xFF)},        // samp_corr_l
            {1, 0x3E, static_cast<std::uint32_t>((offs >> 8) & 0x3F)}, // samp_corr_h
        });
    }

    Result resetBuffer() {
        if (auto r = setUsbReg(kUsbEpaCtl, kEpaCtlReset, 2); !r) {
            return r;
        }
        if (auto r = setUsbReg(kUsbEpaCtl, kEpaCtlRelease, 2); !r) {
            return r;
        }
#if defined(__EMSCRIPTEN__)
        detail::iqQueue().reset();
#endif
        return {};
    }

    // data transfer

    std::expected<std::size_t, std::string> readBulk(std::uint8_t* dst, std::size_t maxLen) {
#if !defined(__EMSCRIPTEN__)
        return _usb.bulkRead(kBulkEndpoint, {dst, maxLen}, 100);
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

    // USB transport (the ONLY platform boundary)

    Result ctrlTransferOut(std::uint16_t wValue, std::uint16_t wIndex, std::span<const std::uint8_t> data) {
#if !defined(__EMSCRIPTEN__)
        return _usb.controlOut(kVendorOut, 0, wValue, wIndex, data, kCtrlTimeout);
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
        return _usb.controlIn(kVendorIn, 0, wValue, wIndex, length, kCtrlTimeout);
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

    // RTL2832U register access (platform-agnostic)

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
        if (auto r = ctrlTransferOut(static_cast<std::uint16_t>((addr << 8) | 0x20), static_cast<std::uint16_t>(page | kWriteFlag), {data.data(), len}); !r) {
            return r;
        }
        return ctrlTransferIn(0x0120, 0x0A, 1).transform([](auto&&) {}); // read-back confirmation
    }

    // I2C access

    Result openI2C() { return setDemodReg(1, 0x01, kI2cRepeaterEnable, 1); }
    Result closeI2C() { return setDemodReg(1, 0x01, kI2cRepeaterDisable, 1); }

    struct I2CGate {
        RTL2832Device* _dev = nullptr;
        I2CGate()           = default;
        explicit I2CGate(RTL2832Device* dev) : _dev(dev) {}
        I2CGate(const I2CGate&)            = delete;
        I2CGate& operator=(const I2CGate&) = delete;
        I2CGate(I2CGate&& o) noexcept : _dev(std::exchange(o._dev, nullptr)) {}
        I2CGate& operator=(I2CGate&&) = delete;
        ~I2CGate() {
            if (_dev) {
                static_cast<void>(_dev->closeI2C());
            }
        }
    };

    [[nodiscard]] std::expected<I2CGate, std::string> i2cGate() {
        if (auto r = openI2C(); !r) {
            return std::unexpected(r.error());
        }
        return I2CGate{this};
    }

    Result setI2CReg(std::uint8_t i2cAddr, std::uint8_t reg, std::uint8_t value) {
        std::array<std::uint8_t, 2> data{reg, value};
        return ctrlTransferOut(i2cAddr, kBlockIic | kWriteFlag, data);
    }

    [[nodiscard]] std::expected<std::uint8_t, std::string> getI2CReg(std::uint8_t i2cAddr, std::uint8_t reg) {
        std::array<std::uint8_t, 1> regData{reg};
        if (auto r = ctrlTransferOut(i2cAddr, kBlockIic | kWriteFlag, regData); !r) {
            return std::unexpected(r.error());
        }
        auto result = ctrlTransferIn(i2cAddr, kBlockIic, 1);
        if (!result) {
            return std::unexpected(result.error());
        }
        return result->empty() ? std::uint8_t{0} : (*result)[0];
    }

    [[nodiscard]] std::expected<std::vector<std::uint8_t>, std::string> getI2CRegBuf(std::uint8_t i2cAddr, std::uint8_t reg, std::uint8_t len) {
        std::array<std::uint8_t, 1> regData{reg};
        if (auto r = ctrlTransferOut(i2cAddr, kBlockIic | kWriteFlag, regData); !r) {
            return std::unexpected(r.error());
        }
        return ctrlTransferIn(i2cAddr, kBlockIic, len);
    }

    // R820T tuner register access

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

    Result writeDemodBatch(std::initializer_list<DemodWrite> writes) {
        for (const auto& w : writes) {
            if (auto r = setDemodReg(w.page, w.addr, w.value, w.len); !r) {
                return r;
            }
        }
        return {};
    }

    Result writeTunerBatch(std::initializer_list<TunerWrite> writes) {
        for (const auto& w : writes) {
            if (auto r = writeTunerRegMask(w.addr, w.value, w.mask); !r) {
                return r;
            }
        }
        return {};
    }

    Result writeE4kBatch(std::initializer_list<E4kWrite> writes) {
        for (const auto& w : writes) {
            if (auto r = writeE4kReg(w.reg, w.value); !r) {
                return r;
            }
        }
        return {};
    }

    [[nodiscard]] static constexpr std::uint8_t nearestIndex(std::span<const std::uint32_t> table, std::uint32_t target) {
        std::uint8_t  best     = 0;
        std::uint32_t bestDiff = UINT32_MAX;
        for (std::size_t i = 0; i < table.size(); ++i) {
            auto diff = target > table[i] ? target - table[i] : table[i] - target;
            if (diff < bestDiff) {
                bestDiff = diff;
                best     = static_cast<std::uint8_t>(i);
            }
        }
        return best;
    }

    // init sequences

    Result initDevice() {
        if (auto r = initDemod(); !r) {
            return r;
        }
        if (auto r = detectTuner(); !r) {
            return r;
        }
        if (_tunerType == TunerType::none) {
            return std::unexpected("no supported tuner found");
        }
        if (_tunerType == TunerType::e4000) {
            return initE4kTuner();
        }
        return initTuner();
    }

    Result initDemod() {
        // USB block init (datasheet section 11)
        if (auto r = setUsbReg(kUsbSysctl, 0x09, 1); !r) {
            return r;
        }
        if (auto r = setUsbReg(kUsbEpaMaxpkt, kEpaMaxpkt512, 2); !r) {
            return r;
        }
        if (auto r = setUsbReg(kUsbEpaCtl, kEpaCtlReset, 2); !r) {
            return r;
        }

        // power on demod (section 10)
        if (auto r = setSysReg(kDemodCtl1, kDemodCtl1Init); !r) {
            return r;
        }
        if (auto r = setSysReg(kDemodCtl, kDemodCtlPowerOn); !r) {
            return r;
        }

        // soft reset via I2C repeater register
        if (auto r = writeDemodBatch({{1, 0x01, 0x14}, {1, 0x01, kI2cRepeaterDisable}}); !r) {
            return r;
        }

        // clear spectrum inversion, DDC shift, and IF registers
        if (auto r = setDemodReg(1, 0x15, 0x00, 1); !r) {
            return r;
        }
        for (std::uint8_t off = 0x16; off <= 0x1B; ++off) {
            if (auto r = setDemodReg(1, off, 0x00, 1); !r) {
                return r;
            }
        }

        // FIR coefficients (p1:0x1C-0x2F)
        for (std::size_t i = 0; i < kFirCoefficients.size(); ++i) {
            if (auto r = setDemodReg(1, static_cast<std::uint8_t>(0x1C + i), kFirCoefficients[i], 1); !r) {
                return r;
            }
        }

        // baseband and datapath configuration
        return writeDemodBatch({
            {0, 0x19, 0x05}, // sdr_ctrl: SDR mode
            {1, 0x93, 0xF0}, // fsm_state_0
            {1, 0x94, 0x0F}, // fsm_state_1
            {1, 0x11, 0x00}, // en_dagc = off
            {1, 0x04, 0x00}, // agc_loop = off
            {0, 0x61, 0x60}, // pid_filt = off
            {0, 0x06, 0x80}, // opt_adc_iq = default
            {1, 0xB1, 0x1B}, // en_bbin: zero-IF + DC cancel + IQ comp
            {0, 0x0D, 0x83}, // clk_out: disable TP_CK0
        });
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

        {
            auto gate = i2cGate();
            if (!gate) {
                return std::unexpected(gate.error());
            }
            for (const auto& [addr, expected, type] : probes) {
                auto id = getI2CReg(addr, 0x00);
                if (id && *id == expected) {
                    _tunerI2cAddr = addr;
                    _tunerType    = type;
                    break;
                }
            }
            // probe E4000 if no R820T/R828D found (chip ID at register 0x02)
            if (_tunerType == TunerType::none) {
                auto id = getI2CReg(kE4kI2cAddr, 0x02);
                if (id && *id == kE4kChipId) {
                    _tunerI2cAddr = kE4kI2cAddr;
                    _tunerType    = TunerType::e4000;
                }
            }
        } // gate closes I2C

        if (_tunerType == TunerType::r820t || _tunerType == TunerType::r828d) {
            // R820T/R828D: switch demod from zero-IF to low-IF mode
            if (auto r = writeDemodBatch({
                    {1, 0xB1, 0x1A}, // en_bbin: disable zero-IF (use IF)
                    {0, 0x08, 0x4D}, // AD_EN_reg: I-ADC only
                    {1, 0x15, 0x01}, // spec_inv: enable spectrum inversion
                });
                !r) {
                return r;
            }
            std::println("[RTL2832] found {} tuner", _tunerType == TunerType::r820t ? "R820T/R820T2/R860" : "R828D");
        } else if (_tunerType == TunerType::e4000) {
            // E4000: zero-IF — keep initDemod() defaults (both ADCs, no spectrum inversion)
            std::println("[RTL2832] found E4000 tuner");
        } else {
            std::println(stderr, "[RTL2832] no supported tuner found");
            return std::unexpected(std::string("no supported tuner found"));
        }
        return {};
    }

    Result initTuner() {
        _shadowRegs = kR8xxInitRegs;

        {
            auto gate = i2cGate();
            if (!gate) {
                return std::unexpected(gate.error());
            }
            for (std::size_t i = 0; i < _shadowRegs.size(); ++i) {
                if (auto r = setI2CReg(_tunerI2cAddr, static_cast<std::uint8_t>(kRegShadowStart + i), _shadowRegs[i]); !r) {
                    return r;
                }
            }

            // tuner electronics init (R820T2 register descriptions, sections 7.4-7.5)
            if (auto r = writeTunerBatch({
                    {0x0C, 0x00, 0x0F}, // R12: clear xtal_check
                    {0x13, 0x03, 0x03}, // R19: VER_NUM
                    {0x1D, 0x00, 0x38}, // R29: PDET1_GAIN = lowest
                    {0x1C, 0x00, 0xF8}, // R28: PDET3_GAIN = lowest
                    {0x06, 0x10, 0x10}, // R6:  FILT_3DB = +3dB
                    {0x1A, 0x30, 0x30}, // R26: PLL_AUTO_CLK = 8kHz
                    {0x1D, 0xE5, 0xC7}, // R29: detect_bw + PDET1_GAIN + PDET2_GAIN
                    {0x1C, 0x24, 0xF8}, // R28: PDET3_GAIN adjust
                    {0x0D, 0x53, 0xFF}, // R13: LNA_VTHH + LNA_VTHL
                    {0x0E, 0x75, 0xFF}, // R14: MIX_VTH_H + MIX_VTH_L
                    {0x05, 0x00, 0x60}, // R5:  PWD_LNA1 = on, cable1 = off
                    {0x06, 0x00, 0x08}, // R6:  cable2_in = off
                    {0x11, 0x38, 0x38}, // R17: cp_cur = auto
                    {0x17, 0x30, 0x30}, // R23: div_buf_cur = 150uA
                    {0x0A, 0x40, 0x60}, // R10: PW_FILT = low current
                    {0x1E, 0x00, 0x60}, // R30: ext_enable = off
                });
                !r) {
                return r;
            }
        } // gate closes I2C

        // filter calibration at 56 MHz (section 10)
        auto calResult = setPll(kPllCalFreq);
        if (!calResult) {
            return std::unexpected(calResult.error());
        }

        auto gate = i2cGate();
        if (!gate) {
            return std::unexpected(gate.error());
        }
        return writeTunerBatch({
            {0x0B, 0x10, 0x10}, // R11: filt_cal_trig = 1
            {0x0B, 0x00, 0x10}, // R11: filt_cal_trig = 0
            {0x0F, 0x00, 0x08}, // R15: filt_cal_clk = off
        });
    }

    // PLL programming (R820T2 Table 1-3, R16/R20-R22, p. 9)

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
        std::expected<std::vector<std::uint8_t>, std::string> regBuf;
        {
            auto gate = i2cGate();
            if (!gate) {
                return std::unexpected(gate.error());
            }
            regBuf = readTunerBuf(0x00, 5);
        }
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

        auto gate = i2cGate();
        if (!gate) {
            return std::unexpected(gate.error());
        }
        if (auto r = writeTunerRegMask(0x10, static_cast<std::uint8_t>(divNum << 5), 0xE0); !r) { // R16: SEL_DIV
            return std::unexpected(r.error());
        }

        double vcoFreq = freq * mixDiv;
        auto   nint    = static_cast<std::uint32_t>(vcoFreq / (2.0 * pllRef));
        double vcoFra  = vcoFreq - 2.0 * pllRef * nint;
        if (nint < 13) {
            nint = 13;
        }

        auto ni = static_cast<std::uint8_t>((nint - 13) / 4);
        auto si = static_cast<std::uint8_t>((nint - 13) % 4);
        if (auto r = writeTunerReg(0x14, static_cast<std::uint8_t>(ni | (si << 6))); !r) { // R20: NI2C + SI2C
            return std::unexpected(r.error());
        }

        // sigma-delta modulator (section 9.3)
        auto sdm = static_cast<std::uint16_t>(std::min(65535.0, 32768.0 * vcoFra / pllRef));
        if (auto r = writeTunerBatch({
                {0x12, static_cast<std::uint8_t>(sdm == 0 ? 0x08 : 0x00), 0x08}, // R18: PW_SDM
                {0x16, static_cast<std::uint8_t>((sdm >> 8) & 0xFF)},            // R22: SDM_IN[15:8]
                {0x15, static_cast<std::uint8_t>(sdm & 0xFF)},                   // R21: SDM_IN[7:0]
            });
            !r) {
            return std::unexpected(r.error());
        }

        // PLL lock check (section 9.5)
        regBuf = readTunerBuf(0x00, 3);
        if (regBuf && !((*regBuf)[2] & 0x40)) {
            if (auto r = writeTunerRegMask(0x12, 0x60, 0xE0); !r) {
                return std::unexpected(r.error()); // R18: increase VCO current
            }
        }
        // gate closes I2C on return

        double actualFreq = (2.0 * pllRef * (nint + static_cast<double>(sdm) / 65536.0)) / mixDiv;
        return actualFreq;
    }

    // mux configuration (section 11)

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

        auto gate = i2cGate();
        if (!gate) {
            return std::unexpected(gate.error());
        }
        return writeTunerBatch({
            {0x17, cfg->openDrain, 0x08}, // R23: OPEN_D
            {0x1A, cfg->rfMuxPoly, 0xC3}, // R26: RFMUX + RFFILT
            {0x1B, cfg->tfC, 0xFF},       // R27: TF_NCH + TF_LP
            {0x10, 0x00, 0x0B},           // R16: CAPX = 0pF
            {0x08, 0x00, 0x3F},           // R8:  IMR_G = 0
            {0x09, 0x00, 0x3F},           // R9:  IMR_P = 0
        });
    }

    // E4000 tuner support (Elonics E4000 datasheet v4.0)

    Result writeE4kReg(std::uint8_t reg, std::uint8_t value) { return setI2CReg(kE4kI2cAddr, reg, value); }

    [[nodiscard]] std::expected<std::uint8_t, std::string> readE4kReg(std::uint8_t reg) { return getI2CReg(kE4kI2cAddr, reg); }

    Result initE4kTuner() {
        {
            auto gate = i2cGate();
            if (!gate) {
                return std::unexpected(gate.error());
            }

            if (auto r = writeE4kBatch({
                    // power-on reset (datasheet section 7.1)
                    {0x00, 0x07},              // MASTER1: reset + standby + POR detect clear
                    {0x05, 0x00},              // CLK_INP: crystal oscillator
                    {0x06, 0x00},              // REF_CLK: internal reference
                    {kE4kRegClkoutPwdn, 0x96}, // disable clock output
                    // analog bias and signal path configuration
                    {0x7E, 0x01}, {0x7F, 0xFE}, {0x82, 0x00}, {0x86, 0x50}, // I/Q signal path polarity
                    {0x87, 0x20}, {0x88, 0x01}, {0x9F, 0x7F}, {0xA0, 0x07},
                    // AGC thresholds and mode (section 8.2)
                    {0x1D, 0x10},        // AGC4: high threshold
                    {0x1E, 0x04},        // AGC5: low threshold
                    {0x1F, 0x1A},        // AGC6: LNA cal request
                    {kE4kRegAgc1, 0x00}, // serial/manual AGC mode
                    {kE4kRegAgc7, 0x00}, // manual mixer gain
                    // default IF gains
                    {kE4kRegGain3, 0x01}, // stage1=6dB
                    {kE4kRegGain4, 0x12}, // stages5-6=9dB each
                    // IF filters: widest bandwidths for SDR use
                    {kE4kRegFilt2, 0x00}, // mixer=27MHz, RC=21.4MHz
                    {kE4kRegFilt3, 0x00}, // channel=5.5MHz, enabled
                    // DC offset: disable LUT, clear time constants
                    {kE4kRegDc5, 0x00}, {0x70, 0x00}, // DCTIME1
                    {0x71, 0x00},                     // DCTIME2
                });
                !r) {
                return r;
            }
        } // gate closes I2C

        if (auto r = calibrateE4kDcOffset(); !r) {
            return r;
        }
        return {};
    }

    // fractional-N PLL programming (datasheet section 9)
    // f_VCO = f_LO × mul,  f_VCO = f_xtal × (Z + X/65536)
    ValueResult setE4kPll(double freq) {
        auto freqKhz = static_cast<std::uint32_t>(freq / 1e3);

        const auto* entry = &kE4kPllLut.back();
        for (const auto& e : kE4kPllLut) {
            if (freqKhz < e.maxFreqKhz) {
                entry = &e;
                break;
            }
        }

        auto xtalKhz   = kXtalFreq / 1000U;
        auto fVcoKhz   = static_cast<std::uint64_t>(freqKhz) * entry->mul;
        auto z         = static_cast<std::uint8_t>(fVcoKhz / xtalKhz);
        auto remainder = fVcoKhz - static_cast<std::uint64_t>(xtalKhz) * z;
        auto x         = static_cast<std::uint16_t>((remainder * 65536ULL) / xtalKhz);

        auto gate = i2cGate();
        if (!gate) {
            return std::unexpected(gate.error());
        }

        if (auto r = writeE4kBatch({
                {kE4kRegSynth7, entry->synth7},
                {kE4kRegSynth3, z},
                {kE4kRegSynth4, static_cast<std::uint8_t>(x & 0xFF)},
                {kE4kRegSynth5, static_cast<std::uint8_t>(x >> 8)},
            });
            !r) {
            return std::unexpected(r.error());
        }

        // frequency band selection (SYNTH1[2:1])
        E4kBand band;
        if (freq < 140e6) {
            band = E4kBand::vhf2;
        } else if (freq < 350e6) {
            band = E4kBand::vhf3;
        } else if (freq < 1135e6) {
            band = E4kBand::uhf;
        } else {
            band = E4kBand::lBand;
        }

        if (auto synth1 = readE4kReg(kE4kRegSynth1)) {
            auto bandBits = static_cast<std::uint8_t>(static_cast<std::uint8_t>(band) << 1);
            if (auto r = writeE4kReg(kE4kRegSynth1, static_cast<std::uint8_t>((*synth1 & ~0x06U) | bandBits)); !r) {
                return std::unexpected(r.error());
            }
        }

        // UHF band requires increased bias current for the VCO buffer
        if (auto r = writeE4kReg(kE4kRegBias, band == E4kBand::uhf ? 0x03 : 0x00); !r) {
            return std::unexpected(r.error());
        }

        // verify PLL lock (SYNTH1[0])
        if (auto lock = readE4kReg(kE4kRegSynth1); lock && !(*lock & 0x01)) {
            std::println(stderr, "[RTL2832] E4000 PLL did not lock at {:.0f} Hz", freq);
        }
        // gate closes I2C on return

        return static_cast<double>(kXtalFreq) * (z + static_cast<double>(x) / 65536.0) / entry->mul;
    }

    // RF input filter band selection (E4000 Table 21, p. 43)
    Result setE4kBandFilter(double freq) {
        auto         freqMhz   = static_cast<std::uint32_t>(freq / 1e6);
        std::uint8_t filterIdx = 0;

        if (freq >= 1135e6) {
            filterIdx = nearestIndex(kE4kRfFilterLbandMhz, freqMhz);
        } else if (freq >= 350e6) {
            filterIdx = nearestIndex(kE4kRfFilterUhfMhz, freqMhz);
        }
        // VHF2/VHF3: fixed filter (index 0)

        auto gate = i2cGate();
        if (!gate) {
            return std::unexpected(gate.error());
        }
        if (auto filt1 = readE4kReg(kE4kRegFilt1)) {
            if (auto r = writeE4kReg(kE4kRegFilt1, static_cast<std::uint8_t>((*filt1 & 0xF0) | (filterIdx & 0x0F))); !r) {
                return r;
            }
        }
        return {};
    }

    // DC offset calibration at each mixer/IF1 gain combination (E4000 sections 1.25-1.27, p. 48-49)
    Result calibrateE4kDcOffset() {
        constexpr std::array<std::pair<std::uint8_t, std::uint8_t>, 4> kGainCombos{{
            {0x00, 0x00}, // mixer=4dB,  IF1=-3dB
            {0x00, 0x01}, // mixer=4dB,  IF1=+6dB
            {0x01, 0x00}, // mixer=12dB, IF1=-3dB
            {0x01, 0x01}, // mixer=12dB, IF1=+6dB
        }};

        auto gate = i2cGate();
        if (!gate) {
            return std::unexpected(gate.error());
        }
        for (std::size_t i = 0; i < kGainCombos.size(); ++i) {
            auto [mixGain, ifGain] = kGainCombos[i];
            if (auto r = writeE4kBatch({{kE4kRegGain2, mixGain}, {kE4kRegGain3, ifGain}, {kE4kRegDc1, 0x01}}); !r) {
                return r;
            }

            auto dcI = readE4kReg(0x2A); // DC2: I offset
            auto dcQ = readE4kReg(0x2B); // DC3: Q offset
            if (dcI && dcQ) {
                auto idx = static_cast<std::uint8_t>(i);
                if (auto r = writeE4kBatch({
                        {static_cast<std::uint8_t>(0x50 + idx), *dcQ}, // QLUT
                        {static_cast<std::uint8_t>(0x60 + idx), *dcI}, // ILUT
                    });
                    !r) {
                    return r;
                }
            }
        }

        // restore default gains after calibration
        return writeE4kBatch({{kE4kRegGain2, 0x01}, {kE4kRegGain3, 0x01}});
    }
};

} // namespace gr::blocks::sdr

#endif // GNURADIO_RTL2832_DEVICE_HPP
