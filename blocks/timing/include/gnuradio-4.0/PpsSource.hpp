#ifndef GNURADIO_PPS_SOURCE_HPP
#define GNURADIO_PPS_SOURCE_HPP

#if !defined(__linux__)
#error "PpsSource requires Linux (clock_nanosleep, adjtimex, /dev/ptpN, /dev/ppsN)"
#endif

#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/timex.h>
#include <unistd.h>

#include <linux/pps.h>
#include <linux/ptp_clock.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <gnuradio-4.0/NMEAParser.hpp> // EmitMode

namespace gr::timing {

GR_REGISTER_BLOCK("gr::timing::PpsSource", gr::timing::PpsSource)

enum class ClockMode : std::uint8_t {
    NTP,   // CLOCK_REALTIME disciplined by ntpd/chrony
    PTP,   // /dev/ptpN via FD_TO_CLOCKID
    TAI,   // CLOCK_TAI (UTC + tai_offset, no leap-second jumps)
    HwPps, // /dev/ppsN — wait for hardware PPS assert event
    Auto   // probe: PTP if /dev/ptp0 readable, else NTP
};

struct KernelDiscipline {
    bool         synchronised  = false; // !(STA_UNSYNC)
    std::int64_t offsetNs      = 0;     // ntp_adjtime tx.offset (converted to ns)
    std::int64_t estErrorNs    = 0;     // tx.esterror (converted to ns)
    std::int64_t maxErrorNs    = 0;     // tx.maxerror (converted to ns)
    std::int64_t freqPpb       = 0;     // tx.freq (scaled ppm → ppb)
    std::int64_t jitterNs      = 0;     // tx.jitter (STA_NANO-aware)
    std::int32_t taiUtcOffsetS = 0;     // tx.tai
    std::uint8_t leapStatus    = 0;     // TIME_OK / TIME_INS / TIME_DEL / TIME_OOP
};

namespace detail {

// kernel's FD_TO_CLOCKID macro, reimplemented as a function to avoid -Wuseless-cast with GCC
inline clockid_t fdToClockId(int fd) { return (~fd << 3) | 3; }

inline KernelDiscipline queryKernelDiscipline() {
    timex tx{};
    tx.modes      = 0; // read-only query
    int leapState = ntp_adjtime(&tx);

    KernelDiscipline d;
    d.synchronised = !(tx.status & STA_UNSYNC);
    d.leapStatus   = static_cast<std::uint8_t>(leapState >= 0 ? leapState : TIME_ERROR);

    bool nanoMode   = (tx.status & STA_NANO) != 0;
    d.offsetNs      = nanoMode ? tx.offset : tx.offset * 1000;
    d.estErrorNs    = tx.esterror * 1000; // esterror is always in microseconds
    d.maxErrorNs    = tx.maxerror * 1000; // maxerror is always in microseconds
    d.jitterNs      = nanoMode ? tx.jitter : tx.jitter * 1000;
    d.freqPpb       = (tx.freq * 1000) >> 16; // freq is in scaled ppm (1 ppm = 1<<16)
    d.taiUtcOffsetS = tx.tai;
    return d;
}

inline std::uint64_t timespecToNs(const timespec& ts) { return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL + static_cast<std::uint64_t>(ts.tv_nsec); }

inline timespec nextSecondBoundary(clockid_t clkId) {
    timespec now{};
    clock_gettime(clkId, &now);
    return {.tv_sec = now.tv_sec + 1, .tv_nsec = 0};
}

inline constexpr std::string_view clockModeName(ClockMode mode) {
    switch (mode) {
    case ClockMode::NTP: return "NTP";
    case ClockMode::PTP: return "PTP";
    case ClockMode::TAI: return "TAI";
    case ClockMode::HwPps: return "HwPps";
    case ClockMode::Auto: return "Auto";
    }
    return "?";
}

inline constexpr std::string_view leapStatusName(std::uint8_t status) {
    switch (status) {
    case TIME_OK: return "OK";
    case TIME_INS: return "insert_leap";
    case TIME_DEL: return "delete_leap";
    case TIME_OOP: return "leap_in_progress";
    case TIME_WAIT: return "leap_occurred";
    default: return "unsynchronised";
    }
}

struct ScopedFd {
    int fd     = -1;
    ScopedFd() = default;
    explicit ScopedFd(int f) : fd(f) {}
    ~ScopedFd() {
        if (fd >= 0) {
            ::close(fd);
        }
    }
    ScopedFd(const ScopedFd&)            = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;
    ScopedFd(ScopedFd&& o) noexcept : fd(std::exchange(o.fd, -1)) {}
    ScopedFd& operator=(ScopedFd&& o) noexcept {
        std::swap(fd, o.fd);
        return *this;
    }
    [[nodiscard]] int release() noexcept { return std::exchange(fd, -1); }
};

} // namespace detail

struct PpsSource : gr::Block<PpsSource> {
    using Description = Doc<R"(PPS timing source using kernel clocks (NTP/PTP/TAI) or hardware PPS (/dev/ppsN).
Emits one sample per UTC second boundary with TRIGGER_TIME and kernel discipline metadata as tags.
Linux only — uses clock_nanosleep, adjtimex, /dev/ptpN, and /dev/ppsN kernel interfaces.)">;

    gr::PortOut<std::uint8_t> out;

    Annotated<ClockMode, "clock_mode", Visible, Doc<"clock source: NTP, PTP, TAI, HwPps, or Auto">>                               clock_mode       = ClockMode::Auto;
    Annotated<std::uint8_t, "ptp_device_index", Visible, Doc<"/dev/ptpN index (PTP mode)">>                                       ptp_device_index = 0;
    Annotated<std::uint8_t, "pps_device_index", Visible, Doc<"/dev/ppsN index (HwPps mode)">>                                     pps_device_index = 0;
    Annotated<std::string, "trigger_name", Doc<"tag trigger name prefix">>                                                        trigger_name     = std::string("PPS");
    Annotated<std::string, "timing context">                                                                                      context;
    Annotated<EmitMode, "emit_mode", Visible, Doc<"ppsOnly: 1 sample/PPS, clock: sample_rate samples/s">>                         emit_mode      = EmitMode::ppsOnly;
    Annotated<float, "sample_rate", Visible, Unit<"Hz">, Doc<"1 Hz in ppsOnly, user-defined in clock mode">, Limits<1.f, 100e6f>> sample_rate    = 1.f;
    Annotated<bool, "emit_meta_info", Doc<"include kernel discipline and clock details in tags">>                                 emit_meta_info = true;

    GR_MAKE_REFLECTABLE(PpsSource, out, clock_mode, ptp_device_index, pps_device_index, trigger_name, context, emit_mode, sample_rate, emit_meta_info);

    ClockMode        _resolvedMode = ClockMode::NTP;
    clockid_t        _clockId      = CLOCK_REALTIME;
    detail::ScopedFd _ptpFd;
    detail::ScopedFd _ppsFd;
    std::uint64_t    _seq          = 0;
    bool             _ioThreadDone = true;

    struct IoThreadGuard {
        bool& done;
        ~IoThreadGuard() { gr::atomic_ref(done).wait(false); }
    };
    IoThreadGuard _ioGuard{_ioThreadDone};

    void start() {
        _seq               = 0;
        auto resolveResult = resolveClockMode();
        if (!resolveResult) {
            std::println(stderr, "[PPS] clock resolution failed: {} — falling back to NTP", resolveResult.error());
            _resolvedMode = ClockMode::NTP;
            _clockId      = CLOCK_REALTIME;
        }
        std::println("[PPS] mode={} (requested={})", detail::clockModeName(_resolvedMode), detail::clockModeName(clock_mode));

        gr::atomic_ref(_ioThreadDone).store_release(false);
        thread_pool::Manager::defaultIoPool()->execute([this]() { ioLoop(); });
    }

    void stop() {
        gr::atomic_ref(_ioThreadDone).wait(false);
        _ptpFd = {};
        _ppsFd = {};
    }

    work::Result work(std::size_t requestedWork = std::numeric_limits<std::size_t>::max()) noexcept {
        if (!lifecycle::isActive(this->state())) {
            return {requestedWork, 0UZ, work::Status::DONE};
        }
        return {requestedWork, 1UZ, work::Status::OK};
    }

    void publishPpsTick(std::uint64_t nominalUtcNs, std::int64_t wakeupOffsetNs, const KernelDiscipline& discipline) {
        std::size_t nSamples = (emit_mode == EmitMode::ppsOnly) ? 1UZ : static_cast<std::size_t>(sample_rate);
        auto        span     = out.streamWriter().template tryReserve<SpanReleasePolicy::ProcessNone>(nSamples);
        if (span.empty()) {
            return;
        }
        std::ranges::fill(span, std::uint8_t{0});

        bool locked    = discipline.synchronised;
        auto triggerNs = nominalUtcNs;
        auto modeName  = std::string(detail::clockModeName(_resolvedMode));

        auto tagMap = out.makeTagMap();
        tag::put(tagMap, tag::TRIGGER_NAME, locked ? std::format("{}_{}", trigger_name.value, modeName) : std::format("{}_{} (unlocked)", trigger_name.value, modeName));
        tag::put(tagMap, tag::TRIGGER_TIME, triggerNs);
        tag::put(tagMap, tag::TRIGGER_OFFSET, 0.f);

        if (emit_meta_info) {
            auto metaInfo = out.makeTagMap();
            tag::put(metaInfo, "clock_mode", modeName);
            tag::put(metaInfo, "wakeup_offset_ns", wakeupOffsetNs);
            tag::put(metaInfo, "sequence", _seq);
            tag::put(metaInfo, "synchronised", discipline.synchronised);
            tag::put(metaInfo, "kernel_offset_ns", discipline.offsetNs);
            tag::put(metaInfo, "est_error_ns", discipline.estErrorNs);
            tag::put(metaInfo, "max_error_ns", discipline.maxErrorNs);
            tag::put(metaInfo, "freq_ppb", discipline.freqPpb);
            tag::put(metaInfo, "jitter_ns", discipline.jitterNs);
            tag::put(metaInfo, "tai_utc_offset_s", discipline.taiUtcOffsetS);
            tag::put(metaInfo, "leap_status", std::string(detail::leapStatusName(discipline.leapStatus)));
            tag::put(tagMap, tag::TRIGGER_META_INFO, std::move(metaInfo));
        }
        if (!context.value.empty()) {
            tag::put(tagMap, tag::CONTEXT, context.value);
        }
        out.publishTag(std::move(tagMap), 0UZ);
        span.publish(span.size());

        this->progress->incrementAndGet();
        this->progress->notify_all();
        ++_seq;
    }

    std::expected<void, std::string> resolveClockMode() {
        ClockMode mode = clock_mode;

        if (mode == ClockMode::Auto) {
            auto ptpPath = std::format("/dev/ptp{}", ptp_device_index);
            int  fd      = ::open(ptpPath.c_str(), O_RDONLY);
            if (fd >= 0) {
                _ptpFd        = detail::ScopedFd(fd);
                _resolvedMode = ClockMode::PTP;
                _clockId      = detail::fdToClockId(_ptpFd.fd);
                return {};
            }
            // fall back to NTP
            _resolvedMode = ClockMode::NTP;
            _clockId      = CLOCK_REALTIME;
            return {};
        }

        if (mode == ClockMode::PTP) {
            auto ptpPath = std::format("/dev/ptp{}", ptp_device_index);
            int  fd      = ::open(ptpPath.c_str(), O_RDONLY);
            if (fd < 0) {
                return std::unexpected(std::format("cannot open {}: {}", ptpPath, std::strerror(errno)));
            }
            _ptpFd        = detail::ScopedFd(fd);
            _resolvedMode = ClockMode::PTP;
            _clockId      = detail::fdToClockId(_ptpFd.fd);
            return {};
        }

        if (mode == ClockMode::HwPps) {
            auto ppsPath = std::format("/dev/pps{}", pps_device_index);
            int  fd      = ::open(ppsPath.c_str(), O_RDONLY);
            if (fd < 0) {
                return std::unexpected(std::format("cannot open {}: {}", ppsPath, std::strerror(errno)));
            }
            _ppsFd        = detail::ScopedFd(fd);
            _resolvedMode = ClockMode::HwPps;
            _clockId      = CLOCK_REALTIME; // wall clock for timestamps
            return {};
        }

        if (mode == ClockMode::TAI) {
            _resolvedMode = ClockMode::TAI;
            _clockId      = CLOCK_TAI;
            return {};
        }

        // NTP (default)
        _resolvedMode = ClockMode::NTP;
        _clockId      = CLOCK_REALTIME;
        return {};
    }

    void ioLoop() {
        thread_pool::thread::setThreadName(std::format("pps:{}", this->name.value));

        while (lifecycle::isActive(this->state())) {
            this->applyChangedSettings();

            if (_resolvedMode == ClockMode::HwPps) {
                waitForHwPps();
            } else {
                waitForClockPps();
            }
        }

        gr::atomic_ref(_ioThreadDone).store_release(true);
        gr::atomic_ref(_ioThreadDone).notify_all();
    }

    void waitForClockPps() {
        auto target = detail::nextSecondBoundary(_clockId);
        int  ret    = clock_nanosleep(_clockId, TIMER_ABSTIME, &target, nullptr);
        if (ret != 0) {
            if (ret == EINTR) {
                return; // interrupted, retry
            }
            std::println(stderr, "[PPS] clock_nanosleep failed: {}", std::strerror(ret));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return;
        }

        timespec actual{};
        clock_gettime(_clockId, &actual);

        auto nominalNs    = detail::timespecToNs(target);
        auto actualNs     = detail::timespecToNs(actual);
        auto wakeupOffset = static_cast<std::int64_t>(actualNs) - static_cast<std::int64_t>(nominalNs);
        auto discipline   = detail::queryKernelDiscipline();

        // for TAI mode, convert nominal to UTC for TRIGGER_TIME
        std::uint64_t triggerUtcNs = nominalNs;
        if (_resolvedMode == ClockMode::TAI && discipline.taiUtcOffsetS > 0) {
            triggerUtcNs -= static_cast<std::uint64_t>(static_cast<std::uint32_t>(discipline.taiUtcOffsetS)) * 1'000'000'000ULL;
        }

        publishPpsTick(triggerUtcNs, wakeupOffset, discipline);
    }

    void waitForHwPps() {
        pps_fdata fetchData{};
        fetchData.timeout = {.sec = 2, .nsec = 0, .flags = 0};

        int ret = ::ioctl(_ppsFd.fd, PPS_FETCH, &fetchData);
        if (ret < 0) {
            if (errno == EINTR) {
                return;
            }
            if (errno == ETIMEDOUT) {
                // no PPS pulse within timeout — emit with wall clock as fallback
                auto now                = detail::wallClockNs();
                auto discipline         = detail::queryKernelDiscipline();
                discipline.synchronised = false;
                publishPpsTick(now, 0, discipline);
                return;
            }
            std::println(stderr, "[PPS] PPS_FETCH failed: {}", std::strerror(errno));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return;
        }

        auto assertTs     = fetchData.info.assert_tu;
        auto nominalNs    = static_cast<std::uint64_t>(assertTs.sec) * 1'000'000'000ULL;
        auto wakeupOffset = static_cast<std::int64_t>(assertTs.nsec);
        auto discipline   = detail::queryKernelDiscipline();

        publishPpsTick(nominalNs, wakeupOffset, discipline);
    }
};

} // namespace gr::timing

#endif // GNURADIO_PPS_SOURCE_HPP
