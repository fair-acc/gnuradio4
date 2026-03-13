#ifndef GNURADIO_GPS_SOURCE_HPP
#define GNURADIO_GPS_SOURCE_HPP

#include <array>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <gnuradio-4.0/NMEADevice.hpp>
#include <gnuradio-4.0/NMEAParser.hpp>

namespace gr::timing {

GR_REGISTER_BLOCK("gr::timing::GpsSource", gr::timing::GpsSource)

struct GpsSource : gr::Block<GpsSource> {
    using Description = Doc<R"(GPS/GNSS serial timing source. Reads NMEA sentences from a serial port (auto-detected
or explicit device_path) on a background IO thread and emits PPS-synchronised samples with
TRIGGER_TIME, geolocation, and satellite metadata as tags. Supports POSIX, Win32, and WASM/WebSerial.

N.B. This block/HW is particularly useful due to it's common sub-us level PPS HW trigger pulse, that can be used
for synchronising otherwise undisciplined SDRs using their PPS)">;

    gr::PortOut<std::uint8_t> out;

    Annotated<std::string, "device_path", Doc<"serial device path, empty for auto-detect">>                                       device_path;
    Annotated<std::string, "device_name", Visible, Doc<"resolved device description, set on open">>                               device_name;
    Annotated<std::string, "trigger_name", Doc<"tag trigger name prefix">>                                                        trigger_name = std::string("GPS_PPS");
    Annotated<std::string, "timing context">                                                                                      context;
    Annotated<EmitMode, "emit_mode", Visible, Doc<"ppsOnly: 1 sample/PPS, clock: sample_rate samples/s">>                         emit_mode        = EmitMode::ppsOnly;
    Annotated<float, "sample_rate", Visible, Unit<"Hz">, Doc<"1 Hz in ppsOnly, user-defined in clock mode">, Limits<1.f, 100e6f>> sample_rate      = 1.f;
    Annotated<bool, "emit_meta_info", Doc<"include geolocation and satellite details in tags">>                                   emit_meta_info   = true;
    Annotated<bool, "emit_device_info", Doc<"include device_name in trigger_meta_info">>                                          emit_device_info = true;
    Annotated<BaudRate, "baud_rate">                                                                                              baud_rate        = BaudRate::Baud9600;
    Annotated<std::uint32_t, "update_rate_ms", Doc<"GPS update interval request [ms]">>                                           update_rate_ms   = 1000U;

    GR_MAKE_REFLECTABLE(GpsSource, out, device_path, device_name, trigger_name, context, emit_mode, sample_rate, emit_meta_info, emit_device_info, baud_rate, update_rate_ms);

    SerialPort _serialPort;
    NMEAParser _parser;
    bool       _ioThreadDone = true; // true until start() launches the IO thread

    struct IoThreadGuard { // must be last member — destroyed first, ensuring IO thread exits before _serialPort/_parser
        bool& done;
        ~IoThreadGuard() { gr::atomic_ref(done).wait(false); }
    };
    IoThreadGuard _ioGuard{_ioThreadDone};

    void start() {
        _parser = NMEAParser{};
        configureDefaultBaudRate(baud_rate);
        gr::atomic_ref(_ioThreadDone).store_release(false);
        thread_pool::Manager::defaultIoPool()->execute([this]() { ioReadLoop(); });
    }

    void stop() {
        gr::atomic_ref(_ioThreadDone).wait(false);
        _serialPort.close();
    }

    work::Result work(std::size_t requestedWork = std::numeric_limits<std::size_t>::max()) noexcept {
        if (!lifecycle::isActive(this->state())) {
            return {requestedWork, 0UZ, work::Status::DONE};
        }
        return {requestedWork, 1UZ, work::Status::OK};
    }

    void publishPps(const GpsFix& fix, std::uint64_t localTimeNs) {
        std::size_t nSamples = (emit_mode == EmitMode::ppsOnly) ? 1UZ : static_cast<std::size_t>(sample_rate);
        auto        span     = out.streamWriter().template tryReserve<SpanReleasePolicy::ProcessNone>(nSamples);
        if (span.empty()) {
            return;
        }
        std::ranges::fill(span, std::uint8_t{0});

        bool locked     = fix.hasTime && fix.fixType != FixType::none;
        auto triggerUtc = locked ? fix.utcTimestampNs : localTimeNs;

        auto tagMap = out.makeTagMap();
        tag::put(tagMap, tag::TRIGGER_NAME, locked ? trigger_name.value : trigger_name.value + "(unlocked)");
        tag::put(tagMap, tag::TRIGGER_TIME, triggerUtc);
        tag::put(tagMap, tag::TRIGGER_OFFSET, 0.f);

        if (emit_meta_info) {
            auto metaInfo = out.makeTagMap();
            auto geoJson  = out.makeTagMap();
            tag::put(geoJson, "type", std::string("Point"));
            tag::put(geoJson, "coordinates", std::vector<float>{fix.longitude, fix.latitude, fix.altitude});
            tag::put(metaInfo, "geolocation", std::move(geoJson));
            tag::put(metaInfo, "local_time", localTimeNs);
            tag::put(metaInfo, "satellites", fix.satellites);
            tag::put(metaInfo, "hdop", fix.hdop);
            tag::put(metaInfo, "fix_type", std::string(magic_enum::enum_name(fix.fixType)));
            tag::put(metaInfo, "speed_kmh", fix.speedKmh);
            tag::put(metaInfo, "heading_deg", fix.headingDeg);
            if (emit_device_info) {
                tag::put(metaInfo, "device_info", device_name.value);
            }
            tag::put(tagMap, tag::TRIGGER_META_INFO, std::move(metaInfo));
        }
        if (!context.value.empty()) {
            tag::put(tagMap, tag::CONTEXT, context.value);
        }
        out.publishTag(std::move(tagMap), 0UZ);
        span.publish(span.size());

        this->progress->incrementAndGet();
        this->progress->notify_all();
    }

    bool tryOpenSerialPort() {
        if (_serialPort.isOpen()) {
            return true;
        }

        std::string deviceToOpen = device_path.value;

        if (deviceToOpen.empty()) {
            auto deviceResult = autoDetectNMEADevice();
            if (!deviceResult) {
                return false;
            }
            deviceToOpen      = deviceResult->devicePath;
            auto& d           = *deviceResult;
            device_name.value = (d.vendorId != 0) ? std::format("{} [VID:{:04x} PID:{:04x}] {} - {}", d.devicePath, d.vendorId, d.productId, d.vendor, d.model) : d.model.empty() ? d.devicePath : std::format("{} - {}", d.devicePath, d.model);
        } else {
            device_name.value = deviceToOpen;
        }

        auto portResult = SerialPort::open(deviceToOpen, baud_rate);
        if (!portResult) {
            return false;
        }
        _serialPort = std::move(*portResult);
        _parser     = NMEAParser{};
        return true;
    }

    void ioReadLoop() {
        thread_pool::thread::setThreadName(std::format("gps:{}", this->name.value));

        std::string             buffer;
        std::array<char, 512UZ> temp{};

        while (lifecycle::isActive(this->state())) {
            this->applyChangedSettings();

            if (!_serialPort.isOpen()) {
                if (!tryOpenSerialPort()) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    continue;
                }
                std::println("[GPS] serial port opened: {}", device_name.value);
            }

            auto n = _serialPort.read(std::span(temp), std::chrono::milliseconds{update_rate_ms});
            if (n > 0) {
                auto localTime = detail::wallClockNs();
                buffer.append(temp.data(), n);

                std::size_t pos;
                while ((pos = buffer.find('\n')) != std::string::npos) {
                    auto lineView = std::string_view(buffer).substr(0, pos);
                    if (!lineView.empty() && lineView.back() == '\r') {
                        lineView.remove_suffix(1);
                    }

                    if (auto fix = _parser.parseLine(lineView, localTime)) {
                        publishPps(*fix, localTime);
                    }
                    buffer.erase(0, pos + 1);
                }
            }
        }

        gr::atomic_ref(_ioThreadDone).store_release(true);
        gr::atomic_ref(_ioThreadDone).notify_all();
    }
};

} // namespace gr::timing

#endif // GNURADIO_GPS_SOURCE_HPP
