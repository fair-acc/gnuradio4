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

    using TSizeChecker = Limits<0U, std::numeric_limits<std::uint32_t>::max(), [](std::uint32_t x) { return std::has_single_bit(x); }>;
    using TBasePort    = PortOut<T>;
    using TPortType    = std::conditional_t<nPorts == 1U, TBasePort, std::conditional_t<nPorts == std::dynamic_extent, std::vector<TBasePort>, std::array<TBasePort, nPorts>>>;

    gr::PortIn<std::uint8_t, Optional> clk_in;
    TPortType                          out;

    Annotated<std::string, "device", Visible, Doc<"SoapySDR driver name">>                                                       device;
    Annotated<std::string, "device_parameter", Visible, Doc<"additional driver parameters">>                                     device_parameter;
    Annotated<float, "sample_rate", Unit<"Hz">, Visible, Doc<"ADC sample rate (SigMF core:sample_rate)">>                        sample_rate  = 1'000'000.f;
    Annotated<gr::Size_t, "num_channels", Visible, Doc<"number of RX channels (SigMF core:num_channels)">>                       num_channels = 1U;
    Annotated<std::vector<std::string>, "rx_antennae", Visible, Doc<"per-channel RX antenna selection">>                         rx_antennae;
    Annotated<std::vector<double>, "frequency", Unit<"Hz">, Visible, Doc<"per-channel center frequency (SigMF core:frequency)">> frequency     = initDefaultValues(107'000'000.);
    Annotated<std::vector<double>, "rx_bandwidths", Unit<"Hz">, Visible, Doc<"per-channel RX RF bandwidth">>                     rx_bandwidths = initDefaultValues(500'000.);
    Annotated<std::vector<double>, "rx_gains", Unit<"dB">, Visible, Doc<"per-channel RX tuner gain">>                            rx_gains      = initDefaultValues(10.);

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

    GR_MAKE_REFLECTABLE(SoapySource, clk_in, out, device, device_parameter, sample_rate, num_channels, rx_antennae, frequency, rx_bandwidths, rx_gains, max_chunk_size, max_time_out_us, max_overflow_count, max_fragment_count, verbose_overflow, trigger_name, emit_timing_tags, emit_meta_info, tag_interval, dc_blocker_enabled, dc_blocker_cutoff, ppm_estimator_cutoff, ppm_tag_threshold);

    soapy::Device                          _device{};
    soapy::Device::Stream<T, SOAPY_SDR_RX> _rxStream{};
    bool                                   _ioThreadDone = true;
    std::atomic<gr::Size_t>                _overFlowCount{0U};
    gr::Size_t                             _fragmentCount    = 0U;
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
    float                                  _ppmLastEmitted = 0.0f;

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
            rebuildDcFilter();
            rebuildRateEstimator();
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
        if (newSettings.contains("dc_blocker_cutoff") || newSettings.contains("dc_blocker_enabled")) {
            rebuildDcFilter();
        }
        if (newSettings.contains("ppm_estimator_cutoff")) {
            rebuildRateEstimator();
        }
    }

    void start() {
        _overFlowCount.store(0U, std::memory_order_relaxed);
        _fragmentCount    = 0U;
        _clockOffsetNs    = 0;
        _clockOffsetValid = false;
        _clockTriggerName.clear();
        _firstEmission  = true;
        _lastTagTimeNs  = 0UL;
        _ppmLastEmitted = 0.0f;
        rebuildDcFilter();
        rebuildRateEstimator();
        reinitDevice();
        if (!_device.get()) {
            return;
        }
        gr::atomic_ref(_ioThreadDone).store_release(false);
        thread_pool::Manager::defaultIoPool()->execute([this]() { ioReadLoop(); });
    }

    void stop() {
        gr::atomic_ref(_ioThreadDone).wait(false);
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
        return {requestedWork, 1UZ, work::Status::OK};
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
                std::memcpy(span.data(), readBuf.data(), nSamples * sizeof(T));

                if constexpr (std::is_same_v<T, std::complex<float>>) {
                    if (dc_blocker_enabled) {
                        applyDcBlocker(span.data(), nSamples);
                    }
                }

                if (emit_timing_tags) {
                    auto intervalNs = static_cast<std::uint64_t>(tag_interval.value * 1e9f);
                    if (intervalNs == 0UL || _lastTagTimeNs == 0UL || (tWallNs - _lastTagTimeNs) >= intervalNs) {
                        emitTimingTag(nSamples, tWallNs);
                        _lastTagTimeNs = tWallNs;
                    }
                }

                span.publish(nSamples);
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
                    continue; // backpressure on any channel — retry entire read
                }
                for (std::size_t ch = 0UZ; ch < outWriters.size(); ++ch) {
                    auto span = outWriters[ch].get().template tryReserve<SpanReleasePolicy::ProcessNone>(nSamples);
                    std::memcpy(span.data(), readBufs[ch].data(), nSamples * sizeof(T));
                    span.publish(nSamples);
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
        std::size_t nTagsConsumed = 0;

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
                if (auto* namePtr = it->second.template get_if<std::pmr::string>()) {
                    if (!namePtr->empty()) {
                        _clockTriggerName = std::string(*namePtr);
                    }
                }
            }
        }

        std::ignore  = tagData.consume(nTagsConsumed);
        auto clkSpan = clkReader.get(nAvailable);
        std::ignore  = clkSpan.consume(nAvailable);
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

    static soapy::Kwargs parseKwargsString(const std::string& str) {
        soapy::Kwargs     kwargs;
        std::stringstream ss(str);
        std::string       keyVal;
        while (std::getline(ss, keyVal, ',')) {
            auto pos = keyVal.find('=');
            if (pos != std::string::npos) {
                kwargs[keyVal.substr(0, pos)] = keyVal.substr(pos + 1);
            }
        }
        return kwargs;
    }

    void reinitDevice() {
        _rxStream.reset();
        auto devKwargs = soapy::Kwargs{{"driver", device.value}};
        if (!device_parameter->empty()) {
            devKwargs.merge(parseKwargsString(device_parameter.value));
        }
        auto devResult = soapy::Device::make(devKwargs);
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

        applySampleRate();
        applyAntenna();
        applyFrequency();
        applyBandwidth();
        applyGain();

        auto        supportedFormats = _device.getStreamFormats(SOAPY_SDR_RX, 0);
        const char* requestedFormat  = soapy::detail::toSoapySDRFormat<T>();
        if (!supportedFormats.empty() && std::ranges::find(supportedFormats, std::string(requestedFormat)) == supportedFormats.end()) {
            this->emitErrorMessage("reinitDevice()", std::format("format '{}' not supported (available: {})", requestedFormat, gr::join(supportedFormats, ", ")));
            this->requestStop();
            return;
        }

        std::vector<gr::Size_t> channelIndices(num_channels);
        std::iota(channelIndices.begin(), channelIndices.end(), gr::Size_t{0});
        auto streamResult = _device.setupStream<T, SOAPY_SDR_RX>(channelIndices);
        if (!streamResult) {
            this->emitErrorMessage("reinitDevice()", std::format("{} (requested: {})", streamResult.error(), requestedFormat));
            this->requestStop();
            return;
        }
        _rxStream = std::move(*streamResult);
        if (auto r = _rxStream.activate(); !r) {
            this->emitErrorMessage("reinitDevice()", r.error());
            this->requestStop();
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
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            double bw = rx_bandwidths->at(std::min(static_cast<std::size_t>(i), rx_bandwidths->size() - 1UZ));
            if (auto r = _device.setBandwidth(SOAPY_SDR_RX, i, bw); !r) {
                this->emitErrorMessage("applyBandwidth()", r.error());
            }
        }
    }

    void applyGain() {
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            double g = rx_gains->at(std::min(static_cast<std::size_t>(i), rx_gains->size() - 1UZ));
            if (auto r = _device.setGain(SOAPY_SDR_RX, i, g); !r) {
                this->emitErrorMessage("applyGain()", r.error());
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
        for (std::size_t i = 0; i < nSamples; ++i) {
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
            auto count = _overFlowCount.fetch_add(1U, std::memory_order_relaxed) + 1U;
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
            _fragmentCount++;
            if (_fragmentCount > max_fragment_count) {
                this->emitErrorMessage("ioReadLoop()", std::format("MORE_FRAGMENTS: {} of max {}", _fragmentCount, max_fragment_count));
                this->requestStop();
            }
        } else {
            _fragmentCount = 0U;
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
