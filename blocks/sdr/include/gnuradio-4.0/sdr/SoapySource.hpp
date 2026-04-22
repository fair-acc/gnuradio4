#ifndef GNURADIO_SOAPY_SOURCE_HPP
#define GNURADIO_SOAPY_SOURCE_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <gnuradio-4.0/algorithm/SampleRateEstimator.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>

#include <gnuradio-4.0/sdr/SoapyRaiiWrapper.hpp>

namespace gr::blocks::sdr {

namespace detail {
inline bool equalWithinOnePercent(const auto& a, const auto& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](double x, double y) { return std::abs(x - y) <= 0.01 * std::max(std::abs(x), std::abs(y)); });
}
inline std::uint64_t wallClockNs() { return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count()); }
} // namespace detail

GR_REGISTER_BLOCK("gr::blocks::sdr::SoapySource", gr::blocks::sdr::SoapySource, ([T], 1UZ), [ uint8_t, int16_t, std::complex<float> ])
GR_REGISTER_BLOCK("gr::blocks::sdr::SoapyDualSource", gr::blocks::sdr::SoapySource, ([T], 2UZ), [ uint8_t, int16_t, std::complex<float> ])

template<typename T, std::size_t nPorts = std::dynamic_extent>
struct SoapySource : Block<SoapySource<T, nPorts>> {
    using Description = Doc<R"(SoapySDR source block for SDR hardware.
Supports single and multi-channel RX via SoapySDR's device-agnostic API.
Uses a dedicated IO thread to decouple USB/hardware latency from the scheduler.

Operating modes:
  clk_in connected: forwards external timing tags (GPS/PPS) with clock-offset interpolation
  clk_in disconnected: emits best-effort wall-clock timestamps on every chunk

Tested with RTL-SDR and LimeSDR drivers.)">;

    using TSizeChecker = Limits<std::uint32_t{1}, std::numeric_limits<std::uint32_t>::max(), [](std::uint32_t x) { return std::has_single_bit(x); }>;
    using TBasePort    = PortOut<T>;
    using TPortType    = std::conditional_t<nPorts == 1U, TBasePort, std::conditional_t<nPorts == std::dynamic_extent, std::vector<TBasePort>, std::array<TBasePort, nPorts>>>;

    gr::PortIn<std::uint8_t, Optional> clk_in;
    TPortType                          out;

    Annotated<std::string, "device", Visible, Doc<"SoapySDR driver name">>                                                       device;
    Annotated<std::string, "device_parameter", Visible, Doc<"additional driver parameters">>                                     device_parameter;
    Annotated<double, "master_clock_rate", Unit<"Hz">, Doc<"device master clock rate (0 = auto, set before sample_rate)">>       master_clock_rate = 0.0;
    Annotated<std::string, "clock_source", Doc<"clock reference source (e.g. internal, external, gpsdo)">>                       clock_source;
    Annotated<float, "sample_rate", Unit<"Hz">, Visible, Doc<"ADC sample rate (SigMF core:sample_rate)">>                        sample_rate  = 1'000'000.f;
    Annotated<gr::Size_t, "num_channels", Visible, Doc<"number of RX channels (SigMF core:num_channels)">>                       num_channels = 1U;
    Annotated<std::vector<std::string>, "rx_antennae", Visible, Doc<"per-channel RX antenna selection">>                         rx_antennae;
    Annotated<std::vector<double>, "frequency", Unit<"Hz">, Visible, Doc<"per-channel center frequency (SigMF core:frequency)">> frequency            = initDefaultValues(107'000'000.);
    Annotated<std::vector<double>, "rx_bandwidths", Unit<"Hz">, Visible, Doc<"per-channel RX RF bandwidth">>                     rx_bandwidths        = initDefaultValues(500'000.);
    Annotated<std::vector<double>, "rx_gains", Unit<"dB">, Visible, Doc<"per-channel RX tuner gain">>                            rx_gains             = initDefaultValues(10.);
    Annotated<bool, "gain_mode", Doc<"enable automatic gain control (AGC)">>                                                     gain_mode            = false;
    Annotated<double, "frequency_correction", Unit<"ppm">, Doc<"crystal oscillator drift compensation">>                         frequency_correction = 0.0;
    Annotated<bool, "dc_offset_mode", Doc<"enable hardware automatic DC offset removal">>                                        dc_offset_mode       = false;
    Annotated<std::vector<double>, "dc_offset", Doc<"manual DC offset correction [I0,Q0,I1,Q1,...] per channel">>                dc_offset;
    Annotated<std::vector<double>, "iq_balance", Doc<"manual IQ balance correction [I0,Q0,I1,Q1,...] per channel">>              iq_balance;
    Annotated<std::string, "time_source", Doc<"PPS/GPS time reference (e.g. external, gpsdo)">>                                  time_source;
    Annotated<double, "reference_clock_rate", Unit<"Hz">, Doc<"reference oscillator rate (0 = auto)">>                           reference_clock_rate = 0.0;
    Annotated<std::string, "stream_args", Doc<"SoapySDR stream kwargs (comma-separated key=value)">>                             stream_args;
    Annotated<std::string, "tune_args", Doc<"per-channel tuning kwargs (comma-separated key=value)">>                            tune_args;
    Annotated<std::string, "frontend_mapping", Doc<"logical-to-physical channel mapping">>                                       frontend_mapping;
    Annotated<std::string, "device_settings", Doc<"device-level settings (comma-separated key=value)">>                          device_settings;

    Annotated<std::uint32_t, "max_chunk_size", Doc<"max samples per read (ideally N x 512)">, Visible, TSizeChecker> max_chunk_size       = 512U << 4U;
    Annotated<std::uint32_t, "max_time_out_us", Unit<"us">, Doc<"SoapySDR polling timeout">>                         max_time_out_us      = 1'000;
    Annotated<gr::Size_t, "max_overflow_count", Doc<"max consecutive overflows before stop (0 = disable)">>          max_overflow_count   = 10U;
    Annotated<gr::Size_t, "max_fragment_count", Doc<"max consecutive fragments before stop (0 = disable)">>          max_fragment_count   = 100U;
    Annotated<bool, "verbose_overflow", Doc<"log each overflow event">>                                              verbose_overflow     = false;
    Annotated<std::string, "trigger_name", Doc<"tag trigger_name for free-running wallclock mode">>                  trigger_name         = std::string("SDR_WALLCLOCK");
    Annotated<bool, "emit_timing_tags", Doc<"emit timing tags on every chunk">>                                      emit_timing_tags     = true;
    Annotated<bool, "emit_meta_info", Doc<"include device/clock metadata in timing tags">>                           emit_meta_info       = true;
    Annotated<float, "tag_interval", Unit<"s">, Doc<"minimum interval between timing tags (0 = every chunk)">>       tag_interval         = 1.0f;
    Annotated<bool, "dc_blocker_enabled", Doc<"IIR high-pass to remove DC offset (complex<float> only)">>            dc_blocker_enabled   = false;
    Annotated<float, "dc_blocker_cutoff", Unit<"Hz">, Doc<"DC blocker high-pass cutoff frequency">>                  dc_blocker_cutoff    = 10.f;
    Annotated<float, "ppm_estimator_cutoff", Unit<"Hz">, Doc<"LP cutoff for sample-rate estimator (0 = disable)">>   ppm_estimator_cutoff = 0.f;
    Annotated<float, "ppm_tag_threshold", Doc<"emit corrected frequency/rate when ppm drift exceeds this">>          ppm_tag_threshold    = 0.1f;

    GR_MAKE_REFLECTABLE(SoapySource, clk_in, out, device, device_parameter, master_clock_rate, clock_source, sample_rate, num_channels, rx_antennae, frequency, rx_bandwidths, rx_gains, gain_mode, frequency_correction, dc_offset_mode, dc_offset, iq_balance, time_source, reference_clock_rate, stream_args, tune_args, frontend_mapping, device_settings, max_chunk_size, max_time_out_us, max_overflow_count, max_fragment_count, verbose_overflow, trigger_name, emit_timing_tags, emit_meta_info, tag_interval, dc_blocker_enabled, dc_blocker_cutoff, ppm_estimator_cutoff, ppm_tag_threshold);

    soapy::Device                          _device{};
    soapy::Device::Stream<T, SOAPY_SDR_RX> _rxStream{};
    soapy::Kwargs                          _devKwargs{};
    bool                                   _ioThreadDone = true;
    std::atomic<gr::Size_t>                _overflowCount{0U};
    std::atomic<gr::Size_t>                _fragmentCount{0U};
    std::int64_t                           _clockOffsetNs    = 0;
    bool                                   _clockOffsetValid = false;
    std::string                            _clockTriggerName;
    bool                                   _firstEmission  = true;
    std::uint64_t                          _lastTagTimeNs  = 0UL;
    float                                  _prevSampleRate = 0.f;
    double                                 _prevFrequency  = 0.0;
    filter::Filter<float>                  _dcFilterI;
    filter::Filter<float>                  _dcFilterQ;
    algorithm::SampleRateEstimator         _rateEstimator;
    float                                  _ppmLastEmitted   = 0.0f;
    bool                                   _clockEosReceived = false;
    std::atomic<bool>                      _ioThreadStarted{false};
    std::atomic<bool>                      _dcFilterDirty{false};
    std::atomic<bool>                      _rateEstimatorDirty{false};

    struct IoThreadGuard {
        bool& done;
        ~IoThreadGuard() { gr::atomic_ref(done).wait(false); }
    };
    IoThreadGuard _ioGuard{_ioThreadDone};

    void settingsChanged(const property_map& /*oldSettings*/, property_map& newSettings, property_map& forwardSettings) {
        if (!_device.get()) {
            return;
        }
        if (newSettings.contains("frequency")) {
            applyFrequency();
            if (!frequency->empty()) {
                forwardSettings.insert_or_assign(std::pmr::string("frequency"), frequency->front());
            }
        }
        if (newSettings.contains("sample_rate")) {
            applySampleRate();
            forwardSettings.insert_or_assign(std::pmr::string("sample_rate"), sample_rate.value);
            _dcFilterDirty.store(true, std::memory_order_release);
            _rateEstimatorDirty.store(true, std::memory_order_release);
        }
        if (newSettings.contains("rx_antennae")) {
            applyAntenna();
        }
        if (newSettings.contains("rx_gains")) {
            applyGain();
        }
        if (newSettings.contains("rx_bandwidths")) {
            applyBandwidth();
        }
        if (newSettings.contains("gain_mode")) {
            applyGainMode();
        }
        if (newSettings.contains("frequency_correction")) {
            applyFrequencyCorrection();
        }
        if (newSettings.contains("dc_offset_mode")) {
            applyDcOffsetMode();
        }
        if (newSettings.contains("dc_offset")) {
            applyDcOffset();
        }
        if (newSettings.contains("iq_balance")) {
            applyIqBalance();
        }
        if (newSettings.contains("device_settings")) {
            applyDeviceSettings();
        }
        if (newSettings.contains("dc_blocker_cutoff") || newSettings.contains("dc_blocker_enabled")) {
            _dcFilterDirty.store(true, std::memory_order_release);
        }
        if (newSettings.contains("ppm_estimator_cutoff")) {
            _rateEstimatorDirty.store(true, std::memory_order_release);
        }
    }

    void start() {
        _overflowCount.store(0U, std::memory_order_relaxed);
        _fragmentCount.store(0U, std::memory_order_relaxed);
        _clockOffsetNs    = 0;
        _clockOffsetValid = false;
        _clockTriggerName.clear();
        _firstEmission    = true;
        _lastTagTimeNs    = 0UL;
        _ppmLastEmitted   = 0.0f;
        _clockEosReceived = false;
        _ioThreadStarted.store(false, std::memory_order_relaxed);
        _dcFilterDirty.store(false, std::memory_order_relaxed);
        _rateEstimatorDirty.store(false, std::memory_order_relaxed);
        rebuildDcFilter();
        rebuildRateEstimator();
        reinitDevice();
        if (!_device.get() || !_rxStream.get()) {
            return;
        }
        soapy::detail::DeviceRegistry::registerActivation(_devKwargs, [this] {
            if (auto r = _rxStream.activate(); !r) {
                this->emitErrorMessage("start()", r.error());
                this->requestStop();
                return;
            }
            _ioThreadStarted.store(true, std::memory_order_release);
            gr::atomic_ref(_ioThreadDone).store_release(false);
            thread_pool::Manager::defaultIoPool()->execute([this]() { ioReadLoop(); });
        });
    }

    void stop() {
        if (_ioThreadStarted.load(std::memory_order_acquire)) {
            gr::atomic_ref(_ioThreadDone).wait(false);
        }
        _rxStream.reset();
        _device.reset();
    }

    work::Result work(std::size_t requestedWork = std::numeric_limits<std::size_t>::max()) noexcept {
        if (!lifecycle::isActive(this->state())) {
            return {requestedWork, 0UZ, work::Status::DONE};
        }
        if (this->disconnect_on_done && this->hasNoDownStreamConnectedChildren()) {
            this->requestStop();
            return {requestedWork, 0UZ, work::Status::DONE};
        }
        if (_ioThreadStarted.load(std::memory_order_acquire) && gr::atomic_ref(_ioThreadDone).load_acquire()) {
            this->requestStop();
            return {requestedWork, 0UZ, work::Status::DONE};
        }
        return {requestedWork, 0UZ, work::Status::OK};
    }

    void ioReadLoop() {
        thread_pool::thread::setThreadName(std::format("soapy:{}", this->name.value));

        constexpr std::size_t kReadSize = 512UZ * 16UZ;
        std::size_t           nCh       = static_cast<std::size_t>(num_channels.value);

        auto& clkReader = clk_in.streamReader();
        auto& clkTagRdr = clk_in.tagReader();

        if constexpr (nPorts == 1U) {
            std::vector<T> readBuf(kReadSize);
            auto&          outWriter = out.streamWriter();

            while (lifecycle::isActive(this->state())) {
                this->applyChangedSettings();
                applyDirtyFlags();

                int       flags   = 0;
                long long time_ns = 0;
                int       ret     = _rxStream.readStream(flags, time_ns, max_time_out_us, std::span<T>(readBuf));

                if (ret == SOAPY_SDR_TIMEOUT) {
                    continue;
                }
                if (ret < 0) {
                    if (!handleStreamError(ret)) {
                        break;
                    }
                    continue;
                }
                if (ret == 0) {
                    continue;
                }

                handleStreamFlags(flags);
                auto nSamples = static_cast<std::size_t>(ret);
                auto tWallNs  = detail::wallClockNs();

                if (ppm_estimator_cutoff > 0.f) {
                    _rateEstimator.update(static_cast<double>(tWallNs) * 1e-9, nSamples);
                }

                drainClockInput(clkReader, clkTagRdr);

                auto span = outWriter.template tryReserve<SpanReleasePolicy::ProcessNone>(nSamples);
                if (span.empty()) {
                    continue;
                }
                auto nCopy = std::min(nSamples, span.size());
                std::memcpy(span.data(), readBuf.data(), nCopy * sizeof(T));

                if constexpr (std::is_same_v<T, std::complex<float>>) {
                    if (dc_blocker_enabled) {
                        applyDcBlocker(span.data(), nCopy);
                    }
                }

                if (emit_timing_tags) {
                    auto intervalNs = static_cast<std::uint64_t>(tag_interval.value * 1e9f);
                    if (intervalNs == 0UL || _lastTagTimeNs == 0UL || (tWallNs - _lastTagTimeNs) >= intervalNs) {
                        emitTimingTag(nCopy, tWallNs);
                        _lastTagTimeNs = tWallNs;
                    }
                }

                span.publish(nCopy);
                this->progress->incrementAndGet();
                this->progress->notify_all();
            }
        } else {
            std::vector<std::vector<T>> readBufs(nCh, std::vector<T>(kReadSize));
            using WriterType = std::remove_reference_t<decltype(out[0].streamWriter())>;
            std::vector<std::reference_wrapper<WriterType>> outWriters;
            for (std::size_t ch = 0UZ; ch < nCh && ch < out.size(); ++ch) {
                outWriters.push_back(std::ref(out[ch].streamWriter()));
            }

            while (lifecycle::isActive(this->state())) {
                this->applyChangedSettings();
                applyDirtyFlags();

                int       flags   = 0;
                long long time_ns = 0;

                std::vector<std::span<T>> spans;
                spans.reserve(nCh);
                for (auto& buf : readBufs) {
                    spans.push_back(std::span<T>(buf));
                }
                int ret = _rxStream.readStreamIntoBufferList(flags, time_ns, static_cast<long>(max_time_out_us.value), spans);

                if (ret == SOAPY_SDR_TIMEOUT) {
                    continue;
                }
                if (ret < 0) {
                    if (!handleStreamError(ret)) {
                        break;
                    }
                    continue;
                }
                if (ret == 0) {
                    continue;
                }

                handleStreamFlags(flags);
                auto nSamples = static_cast<std::size_t>(ret);
                auto tWallNs  = detail::wallClockNs();

                if (ppm_estimator_cutoff > 0.f) {
                    _rateEstimator.update(static_cast<double>(tWallNs) * 1e-9, nSamples);
                }

                drainClockInput(clkReader, clkTagRdr);

                bool allAvailable = std::ranges::all_of(outWriters, [nSamples](auto& w) { return w.get().available() >= nSamples; });
                if (!allAvailable) {
                    continue;
                }

                using OutSpanType = decltype(outWriters[0].get().template tryReserve<SpanReleasePolicy::ProcessNone>(0UZ));
                std::vector<OutSpanType> outSpans;
                outSpans.reserve(outWriters.size());
                bool allReserved = true;
                for (auto& w : outWriters) {
                    outSpans.push_back(w.get().template tryReserve<SpanReleasePolicy::ProcessNone>(nSamples));
                    if (outSpans.back().empty()) {
                        allReserved = false;
                        break;
                    }
                }
                if (!allReserved) {
                    outSpans.clear();
                    continue;
                }
                for (std::size_t ch = 0UZ; ch < outSpans.size(); ++ch) {
                    auto nCopy = std::min(nSamples, outSpans[ch].size());
                    std::memcpy(outSpans[ch].data(), readBufs[ch].data(), nCopy * sizeof(T));
                    outSpans[ch].publish(nCopy);
                }

                if (emit_timing_tags) {
                    auto intervalNs = static_cast<std::uint64_t>(tag_interval.value * 1e9f);
                    if (intervalNs == 0UL || _lastTagTimeNs == 0UL || (tWallNs - _lastTagTimeNs) >= intervalNs) {
                        emitTimingTag(nSamples, tWallNs);
                        _lastTagTimeNs = tWallNs;
                    }
                }

                this->progress->incrementAndGet();
                this->progress->notify_all();
            }
        }

        this->publishEoS();
        if (clk_in.isConnected()) {
            std::ignore = clk_in.disconnect();
        }

        if (auto r = _rxStream.deactivate(); !r) {
            this->emitErrorMessage("ioReadLoop()", r.error());
        }

        gr::atomic_ref(_ioThreadDone).store_release(true);
        gr::atomic_ref(_ioThreadDone).notify_all();
    }

    void drainClockInput(auto& clkReader, auto& clkTagRdr) {
        if (!clk_in.isConnected()) {
            return;
        }
        auto nAvailable = clkReader.available();
        if (nAvailable == 0) {
            return;
        }

        auto        tagData       = clkTagRdr.get(clkTagRdr.available());
        std::size_t nTagsConsumed = 0UZ;

        for (const auto& clkTag : tagData) {
            ++nTagsConsumed;
            if (auto it = clkTag.map.find(std::pmr::string(tag::TRIGGER_TIME.shortKey())); it != clkTag.map.end()) {
                if (auto* timePtr = it->second.template get_if<std::uint64_t>()) {
                    auto triggerUtcNs = static_cast<std::int64_t>(*timePtr);
                    _clockOffsetNs    = triggerUtcNs - static_cast<std::int64_t>(detail::wallClockNs());
                    _clockOffsetValid = true;
                }
            }
            if (auto it = clkTag.map.find(std::pmr::string(tag::TRIGGER_NAME.shortKey())); it != clkTag.map.end()) {
                if (auto nameView = it->second.template get_if<std::string_view>()) {
                    if (!nameView->empty()) {
                        _clockTriggerName = std::string(*nameView);
                    }
                }
            }
            static const std::pmr::string kEosKey(tag::END_OF_STREAM.shortKey());
            if (clkTag.map.contains(kEosKey)) {
                _clockEosReceived = true;
            }
        }

        std::ignore  = tagData.consume(nTagsConsumed);
        auto clkSpan = clkReader.get(nAvailable);
        std::ignore  = clkSpan.consume(nAvailable);

        if (_clockEosReceived) {
            this->requestStop();
        }
    }

    void emitTimingTag(std::size_t nSamples, std::uint64_t tWallNs) {
        auto tUtcLastNs      = static_cast<std::int64_t>(tWallNs) + _clockOffsetNs;
        auto chunkDurationNs = static_cast<std::int64_t>(static_cast<double>(nSamples - 1) / static_cast<double>(sample_rate.value) * 1e9);
        auto tUtcFirstNs     = static_cast<std::uint64_t>(std::max(std::int64_t{0}, tUtcLastNs - chunkDurationNs));

        bool hasClock = clk_in.isConnected() && !_clockTriggerName.empty();

        auto publishTag = [&](auto& port) {
            auto tagMap = port.makeTagMap();
            tag::put(tagMap, tag::TRIGGER_NAME, hasClock ? _clockTriggerName : trigger_name.value);
            tag::put(tagMap, tag::TRIGGER_TIME, tUtcFirstNs);
            tag::put(tagMap, tag::TRIGGER_OFFSET, 0.f);

            if (emit_meta_info) {
                auto metaInfo = port.makeTagMap();
                tag::put(metaInfo, "trigger_source", std::string("SoapySource"));

                std::string clockSource = (_clockOffsetValid && !_clockTriggerName.empty()) ? _clockTriggerName : std::string("wallclock");
                tag::put(metaInfo, "clock_source", std::move(clockSource));

                if (_clockOffsetValid) {
                    tag::put(metaInfo, "clock_offset_ns", _clockOffsetNs);
                }

                emitChangedParams(metaInfo);
                tag::put(tagMap, tag::TRIGGER_META_INFO, std::move(metaInfo));
            }

            if (_rateEstimator._initialised && ppm_estimator_cutoff > 0.f) {
                float ppmNow = _rateEstimator.estimatedPpm();
                tag::put(tagMap, "sample_rate", static_cast<float>(_rateEstimator.estimatedRate()));
                if (!frequency->empty()) {
                    tag::put(tagMap, "frequency", frequency->front() * (1.0 + static_cast<double>(ppmNow) * 1e-6));
                }
                tag::put(tagMap, "ppm_error", ppmNow);
                _ppmLastEmitted = ppmNow;
            }

            port.publishTag(std::move(tagMap), 0UZ);
        };

        if constexpr (nPorts == 1U) {
            publishTag(out);
        } else {
            for (std::size_t ch = 0UZ; ch < out.size(); ++ch) {
                publishTag(out[ch]);
            }
        }
    }

    void emitChangedParams(property_map& metaInfo) {
        if (_firstEmission || _prevSampleRate != sample_rate.value) {
            tag::put(metaInfo, "sample_rate", static_cast<double>(sample_rate.value));
            _prevSampleRate = sample_rate.value;
        }
        if (_firstEmission || (!frequency->empty() && _prevFrequency != frequency->front())) {
            if (!frequency->empty()) {
                tag::put(metaInfo, "frequency", frequency->front());
                _prevFrequency = frequency->front();
            }
        }
        _firstEmission = false;
    }

    void reinitDevice() {
        _rxStream.reset();
        _devKwargs = soapy::Kwargs{{"driver", device.value}};
        if (!device_parameter->empty()) {
            _devKwargs.merge(soapy::parseKwargsString(device_parameter.value));
        }
        auto devResult = soapy::Device::make(_devKwargs);
        if (!devResult) {
            this->emitErrorMessage("reinitDevice()", devResult.error());
            this->requestStop();
            return;
        }
        _device = std::move(*devResult);

        std::size_t nChannelMax    = _device.getNumChannels(SOAPY_SDR_RX);
        std::size_t nChannelNeeded = (nPorts != std::dynamic_extent) ? nPorts : static_cast<std::size_t>(num_channels.value);
        if (nChannelMax < nChannelNeeded) {
            this->emitErrorMessage("reinitDevice()", std::format("channel mismatch: need {} but device has {}", nChannelNeeded, nChannelMax));
            this->requestStop();
            return;
        }

        applyClockConfig();
        applySampleRate();
        applyAntenna();
        applyFrequency();
        applyBandwidth();
        applyGain();
        applyGainMode();
        applyFrequencyCorrection();
        applyDcOffsetMode();
        applyDcOffset();
        applyIqBalance();
        applyFrontendMapping();
        applyDeviceSettings();

        auto        supportedFormats = _device.getStreamFormats(SOAPY_SDR_RX, 0);
        const char* requestedFormat  = soapy::detail::toSoapySDRFormat<T>();
        if (!supportedFormats.empty() && std::ranges::find(supportedFormats, std::string(requestedFormat)) == supportedFormats.end()) {
            this->emitErrorMessage("reinitDevice()", std::format("format '{}' not supported (available: {})", requestedFormat, gr::join(supportedFormats, ", ")));
            this->requestStop();
            return;
        }

        std::vector<gr::Size_t> channelIndices(num_channels);
        std::iota(channelIndices.begin(), channelIndices.end(), gr::Size_t{0});
        soapy::Kwargs parsedStreamArgs = stream_args->empty() ? soapy::Kwargs{} : soapy::parseKwargsString(stream_args.value);
        auto          streamResult     = _device.setupStream<T, SOAPY_SDR_RX>(channelIndices, parsedStreamArgs);
        if (!streamResult) {
            this->emitErrorMessage("reinitDevice()", std::format("{} (requested: {})", streamResult.error(), requestedFormat));
            this->requestStop();
            return;
        }
        _rxStream = std::move(*streamResult);
    }

    void applyClockConfig() {
        if (!clock_source->empty()) {
            if (auto r = _device.setClockSource(clock_source.value); !r) {
                this->emitErrorMessage("applyClockConfig()", r.error());
            }
        }
        if (!time_source->empty()) {
            if (auto r = _device.setTimeSource(time_source.value); !r) {
                this->emitErrorMessage("applyClockConfig()", r.error());
            }
        }
        if (master_clock_rate > 0.0) {
            if (auto r = _device.setMasterClockRate(master_clock_rate); !r) {
                this->emitErrorMessage("applyClockConfig()", r.error());
            }
        }
        if (reference_clock_rate > 0.0) {
            if (auto r = _device.setReferenceClockRate(reference_clock_rate); !r) {
                this->emitErrorMessage("applyClockConfig()", r.error());
            }
        }
    }

    void applySampleRate() {
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            if (auto r = _device.setSampleRate(SOAPY_SDR_RX, i, static_cast<double>(sample_rate)); !r) {
                this->emitErrorMessage("applySampleRate()", r.error());
            }
        }
        std::vector<double> actual;
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            actual.push_back(_device.getSampleRate(SOAPY_SDR_RX, i));
        }
        if (!detail::equalWithinOnePercent(actual, std::vector<double>(num_channels, static_cast<double>(sample_rate)))) {
            this->emitErrorMessage("applySampleRate()", std::format("mismatch: set {} vs actual {}", sample_rate, gr::join(actual, ", ")));
        }
    }

    void applyAntenna() {
        if (rx_antennae->empty()) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            const auto& ant = rx_antennae->at(std::min(static_cast<std::size_t>(i), rx_antennae->size() - 1UZ));
            if (!ant.empty()) {
                if (auto r = _device.setAntenna(SOAPY_SDR_RX, i, ant); !r) {
                    this->emitErrorMessage("applyAntenna()", r.error());
                }
            }
        }
    }

    void applyFrequency() {
        if (frequency->empty()) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            double freq = frequency->at(std::min(static_cast<std::size_t>(i), frequency->size() - 1UZ));
            if (auto r = _device.setCenterFrequency(SOAPY_SDR_RX, i, freq); !r) {
                this->emitErrorMessage("applyFrequency()", r.error());
            }
        }
        std::vector<double> actual;
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            actual.push_back(_device.getCenterFrequency(SOAPY_SDR_RX, i));
        }
        if (!detail::equalWithinOnePercent(actual, frequency.value)) {
            this->emitErrorMessage("applyFrequency()", std::format("mismatch: set {} vs actual {}", gr::join(frequency.value, ", "), gr::join(actual, ", ")));
        }
    }

    void applyBandwidth() {
        if (rx_bandwidths->empty()) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            double bw = rx_bandwidths->at(std::min(static_cast<std::size_t>(i), rx_bandwidths->size() - 1UZ));
            if (auto r = _device.setBandwidth(SOAPY_SDR_RX, i, bw); !r) {
                this->emitErrorMessage("applyBandwidth()", r.error());
            }
        }
    }

    void applyGain() {
        if (rx_gains->empty()) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            double g = rx_gains->at(std::min(static_cast<std::size_t>(i), rx_gains->size() - 1UZ));
            if (auto r = _device.setGain(SOAPY_SDR_RX, i, g); !r) {
                this->emitErrorMessage("applyGain()", r.error());
            }
        }
    }

    void applyGainMode() {
        if (!gain_mode) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            if (!_device.hasAutomaticGainControl(SOAPY_SDR_RX, i)) {
                continue;
            }
            if (auto r = _device.setAutomaticGainControl(SOAPY_SDR_RX, i, gain_mode); !r) {
                this->emitErrorMessage("applyGainMode()", r.error());
            }
        }
    }

    void applyFrequencyCorrection() {
        if (frequency_correction == 0.0) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            if (!_device.hasFrequencyCorrection(SOAPY_SDR_RX, i)) {
                continue;
            }
            if (auto r = _device.setFrequencyCorrection(SOAPY_SDR_RX, i, frequency_correction); !r) {
                this->emitErrorMessage("applyFrequencyCorrection()", r.error());
            }
        }
    }

    void applyDcOffsetMode() {
        if (!dc_offset_mode) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            if (!_device.hasDCOffsetMode(SOAPY_SDR_RX, i)) {
                continue;
            }
            if (auto r = _device.setDCOffsetMode(SOAPY_SDR_RX, i, dc_offset_mode); !r) {
                this->emitErrorMessage("applyDcOffsetMode()", r.error());
            }
        }
    }

    void applyDcOffset() {
        if (dc_offset->empty()) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            if (!_device.hasDCOffset(SOAPY_SDR_RX, i)) {
                continue;
            }
            auto idx = static_cast<std::size_t>(i) * 2UZ;
            if (idx + 1UZ >= dc_offset->size()) {
                break;
            }
            if (auto r = _device.setDCOffset(SOAPY_SDR_RX, i, dc_offset->at(idx), dc_offset->at(idx + 1UZ)); !r) {
                this->emitErrorMessage("applyDcOffset()", r.error());
            }
        }
    }

    void applyIqBalance() {
        if (iq_balance->empty()) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            if (!_device.hasIQBalance(SOAPY_SDR_RX, i)) {
                continue;
            }
            auto idx = static_cast<std::size_t>(i) * 2UZ;
            if (idx + 1UZ >= iq_balance->size()) {
                break;
            }
            if (auto r = _device.setIQBalance(SOAPY_SDR_RX, i, iq_balance->at(idx), iq_balance->at(idx + 1UZ)); !r) {
                this->emitErrorMessage("applyIqBalance()", r.error());
            }
        }
    }

    void applyFrontendMapping() {
        if (!frontend_mapping->empty()) {
            _device.setFrontendMapping(SOAPY_SDR_RX, frontend_mapping.value);
        }
    }

    void applyDeviceSettings() {
        if (device_settings->empty()) {
            return;
        }
        for (const auto& [key, value] : soapy::parseKwargsString(device_settings.value)) {
            if (auto r = _device.writeSetting(key, value); !r) {
                this->emitErrorMessage("applyDeviceSettings()", r.error());
            }
        }
    }

    void rebuildDcFilter() {
        if constexpr (std::is_same_v<T, std::complex<float>>) {
            if (dc_blocker_enabled && dc_blocker_cutoff > 0.f && sample_rate > 0.f) {
                auto coeffs = filter::iir::designFilter<float>(filter::Type::HIGHPASS, filter::FilterParameters{.order = 2UZ, .fHigh = static_cast<double>(dc_blocker_cutoff), .fs = static_cast<double>(sample_rate)}, filter::iir::Design::BUTTERWORTH);
                _dcFilterI  = filter::Filter<float>(coeffs);
                _dcFilterQ  = filter::Filter<float>(coeffs);
            }
        }
    }

    void rebuildRateEstimator() {
        if (ppm_estimator_cutoff > 0.f) {
            double nomRate                  = static_cast<double>(sample_rate.value);
            double updateHz                 = (nomRate > 0.0) ? (nomRate / (static_cast<double>(max_chunk_size.value))) : 250.0;
            _rateEstimator.filter_cutoff_hz = ppm_estimator_cutoff;
            _rateEstimator.reset(nomRate, updateHz);
        }
    }

    void applyDcBlocker(std::complex<float>* samples, std::size_t nSamples) {
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            float filteredI = _dcFilterI.processOne(samples[i].real());
            float filteredQ = _dcFilterQ.processOne(samples[i].imag());
            samples[i]      = {filteredI, filteredQ};
        }
    }

    void emitOverflowTag() {
        if constexpr (nPorts == 1U) {
            auto tagMap = out.makeTagMap();
            tag::put(tagMap, "rx_overflow", true);
            out.publishTag(std::move(tagMap), 0UZ);
        } else {
            for (std::size_t ch = 0UZ; ch < out.size(); ++ch) {
                auto tagMap = out[ch].makeTagMap();
                tag::put(tagMap, "rx_overflow", true);
                out[ch].publishTag(std::move(tagMap), 0UZ);
            }
        }
    }

    bool handleStreamError(int ret) {
        switch (ret) {
        case SOAPY_SDR_OVERFLOW: {
            auto count = _overflowCount.fetch_add(1U, std::memory_order_relaxed) + 1U;
            if (verbose_overflow) {
                std::println(stderr, "[SoapySource] OVERFLOW #{}", count);
            }
            emitOverflowTag();
            if (max_overflow_count > 0 && count >= max_overflow_count) {
                this->emitErrorMessage("ioReadLoop()", std::format("OVERFLOW: {} of max {}", count, max_overflow_count));
                this->requestStop();
                return false;
            }
            if (auto r = _rxStream.deactivate(); !r) {
                this->emitErrorMessage("ioReadLoop()", r.error());
            }
            if (auto r = _rxStream.activate(); !r) {
                this->emitErrorMessage("ioReadLoop()", r.error());
                this->requestStop();
                return false;
            }
            return true;
        }
        case SOAPY_SDR_CORRUPTION:
            this->emitErrorMessage("ioReadLoop()", "CORRUPTION");
            this->requestStop();
            return false;
        default:
            this->emitErrorMessage("ioReadLoop()", std::format("stream error: {}", ret));
            this->requestStop();
            return false;
        }
    }

    void handleStreamFlags(int flags) {
        if (max_fragment_count > 0 && (flags & SOAPY_SDR_MORE_FRAGMENTS)) {
            auto count = _fragmentCount.fetch_add(1U, std::memory_order_relaxed) + 1U;
            if (count > max_fragment_count) {
                this->emitErrorMessage("ioReadLoop()", std::format("MORE_FRAGMENTS: {} of max {}", count, max_fragment_count));
                this->requestStop();
            }
        } else {
            _fragmentCount.store(0U, std::memory_order_relaxed);
        }
    }

    void applyDirtyFlags() {
        if (_dcFilterDirty.exchange(false, std::memory_order_acquire)) {
            rebuildDcFilter();
        }
        if (_rateEstimatorDirty.exchange(false, std::memory_order_acquire)) {
            rebuildRateEstimator();
        }
    }

    template<typename U>
    static std::vector<U> initDefaultValues(U initialValue) {
        if constexpr (nPorts != std::dynamic_extent) {
            return std::vector<U>(nPorts, initialValue);
        } else {
            return std::vector<U>(1, initialValue);
        }
    }
};

template<typename T>
using SoapySimpleSource = SoapySource<T, 1UZ>;
template<typename T>
using SoapyDualSource = SoapySource<T, 2UZ>;

} // namespace gr::blocks::sdr

#endif // GNURADIO_SOAPY_SOURCE_HPP
