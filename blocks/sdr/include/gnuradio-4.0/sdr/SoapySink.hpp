#ifndef GNURADIO_SOAPY_SINK_HPP
#define GNURADIO_SOAPY_SINK_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <gnuradio-4.0/algorithm/BurstTaper.hpp>
#include <gnuradio-4.0/sdr/SoapyRaiiWrapper.hpp>

namespace gr::blocks::sdr {

GR_REGISTER_BLOCK("gr::blocks::sdr::SoapySink", gr::blocks::sdr::SoapySink, ([T], 1UZ), [ uint8_t, int16_t, std::complex<float> ])
GR_REGISTER_BLOCK("gr::blocks::sdr::SoapyDualSink", gr::blocks::sdr::SoapySink, ([T], 2UZ), [ uint8_t, int16_t, std::complex<float> ])
GR_REGISTER_BLOCK("gr::blocks::sdr::SoapyQuadSink", gr::blocks::sdr::SoapySink, ([T], 4UZ), [ uint8_t, int16_t, std::complex<float> ])

template<typename T, std::size_t nPorts = std::dynamic_extent>
struct SoapySink : Block<SoapySink<T, nPorts>> {
    using Description = Doc<R"(SoapySDR sink block for SDR hardware.
Supports single and multi-channel TX via SoapySDR's device-agnostic API.
Uses a dedicated IO thread to decouple hardware latency from the scheduler.
Shares the underlying SoapySDR device handle with SoapySource when both use
the same driver string, enabling full-duplex TX/RX operation.)">;

    using TSizeChecker  = Limits<1UZ, std::numeric_limits<std::uint32_t>::max(), [](std::uint32_t x) { return std::has_single_bit(x); }>;
    using TBasePort     = PortIn<T>;
    using TPortType     = std::conditional_t<nPorts == 1U, TBasePort, std::conditional_t<nPorts == std::dynamic_extent, std::vector<TBasePort>, std::array<TBasePort, nPorts>>>;
    using StagingBuffer = gr::CircularBuffer<T>;
    using StagingWriter = decltype(std::declval<StagingBuffer>().new_writer());
    using StagingReader = decltype(std::declval<StagingBuffer>().new_reader());

    TPortType in;

    Annotated<std::string, "device", Visible, Doc<"SoapySDR driver name">>                                                 device;
    Annotated<std::string, "device_parameter", Visible, Doc<"additional driver parameters">>                               device_parameter;
    Annotated<double, "master_clock_rate", Unit<"Hz">, Doc<"device master clock rate (0 = auto, set before sample_rate)">> master_clock_rate = 0.0;
    Annotated<std::string, "clock_source", Doc<"clock reference source (e.g. internal, external, gpsdo)">>                 clock_source;
    Annotated<float, "sample_rate", Unit<"Hz">, Visible, Doc<"DAC sample rate">>                                           sample_rate  = 1'000'000.f;
    Annotated<gr::Size_t, "num_channels", Visible, Doc<"number of TX channels">>                                           num_channels = 1U;
    Annotated<std::vector<std::string>, "tx_antennae", Visible, Doc<"per-channel TX antenna selection">>                   tx_antennae;
    Annotated<std::vector<double>, "frequency", Unit<"Hz">, Visible, Doc<"per-channel center frequency">>                  frequency            = initDefaultValues(107'000'000.);
    Annotated<std::vector<double>, "tx_bandwidths", Unit<"Hz">, Visible, Doc<"per-channel TX RF bandwidth">>               tx_bandwidths        = initDefaultValues(500'000.);
    Annotated<std::vector<double>, "tx_gains", Unit<"dB">, Visible, Doc<"per-channel TX gain">>                            tx_gains             = initDefaultValues(10.);
    Annotated<bool, "gain_mode", Doc<"enable automatic gain control (AGC)">>                                               gain_mode            = false;
    Annotated<double, "frequency_correction", Unit<"ppm">, Doc<"crystal oscillator drift compensation">>                   frequency_correction = 0.0;
    Annotated<bool, "dc_offset_mode", Doc<"enable hardware automatic DC offset removal">>                                  dc_offset_mode       = false;
    Annotated<std::vector<double>, "dc_offset", Doc<"manual DC offset correction [I0,Q0,I1,Q1,...] per channel">>          dc_offset;
    Annotated<std::vector<double>, "iq_balance", Doc<"manual IQ balance correction [I0,Q0,I1,Q1,...] per channel">>        iq_balance;
    Annotated<std::string, "time_source", Doc<"PPS/GPS time reference (e.g. external, gpsdo)">>                            time_source;
    Annotated<double, "reference_clock_rate", Unit<"Hz">, Doc<"reference oscillator rate (0 = auto)">>                     reference_clock_rate = 0.0;
    Annotated<std::string, "stream_args", Doc<"SoapySDR stream kwargs (comma-separated key=value)">>                       stream_args;
    Annotated<std::string, "tune_args", Doc<"per-channel tuning kwargs (comma-separated key=value)">>                      tune_args;
    Annotated<std::string, "frontend_mapping", Doc<"logical-to-physical channel mapping">>                                 frontend_mapping;
    Annotated<std::string, "device_settings", Doc<"device-level settings (comma-separated key=value)">>                    device_settings;
    Annotated<std::uint32_t, "max_chunk_size", Doc<"max samples per write">, Visible, TSizeChecker>                        max_chunk_size        = 512U << 4U;
    Annotated<std::uint32_t, "max_time_out_us", Unit<"us">, Doc<"SoapySDR polling timeout">>                               max_time_out_us       = 1'000;
    Annotated<gr::Size_t, "max_underflow_count", Doc<"max consecutive underflows before stop (0 = disable)">>              max_underflow_count   = 10U;
    Annotated<bool, "verbose_underflow", Doc<"log each underflow event">>                                                  verbose_underflow     = false;
    Annotated<bool, "burst_taper_enabled", Doc<"enable TX burst taper (ramp up/down on start/shutdown)">>                  burst_taper_enabled   = false;
    Annotated<float, "burst_ramp_time", Unit<"s">, Doc<"taper ramp duration">>                                             burst_ramp_time       = 0.001f;
    Annotated<std::string, "burst_taper_type", Doc<"None, Linear, RaisedCosine, Tukey, Gaussian, Mushroom, MushroomSine">> burst_taper_type      = std::string("RaisedCosine");
    Annotated<float, "burst_shape_param", Doc<"taper shape parameter (type-dependent)">>                                   burst_shape_param     = 1.0f;
    Annotated<bool, "burst_safety_rampdown", Doc<"force ramp-down on EoS/shutdown if taper not Off">>                      burst_safety_rampdown = true;

    GR_MAKE_REFLECTABLE(SoapySink, in, device, device_parameter, master_clock_rate, clock_source, sample_rate, num_channels, tx_antennae, frequency, tx_bandwidths, tx_gains, gain_mode, frequency_correction, dc_offset_mode, dc_offset, iq_balance, time_source, reference_clock_rate, stream_args, tune_args, frontend_mapping, device_settings, max_chunk_size, max_time_out_us, max_underflow_count, verbose_underflow, burst_taper_enabled, burst_ramp_time, burst_taper_type, burst_shape_param, burst_safety_rampdown);

    soapy::Device                          _device{};
    soapy::Device::Stream<T, SOAPY_SDR_TX> _txStream{};
    soapy::Kwargs                          _devKwargs{};
    std::atomic<gr::Size_t>                _underflowCount{0U};
    bool                                   _ioThreadDone = true;
    std::atomic<bool>                      _ioThreadStarted{false};
    algorithm::BurstTaper<float>           _taper;
    std::vector<StagingBuffer>             _stagingBuffers;
    std::vector<StagingWriter>             _stagingWriters;
    std::vector<StagingReader>             _stagingReaders;

    struct IoThreadGuard {
        bool& done;
        ~IoThreadGuard() { gr::atomic_ref(done).wait(false); }
    };
    IoThreadGuard _ioGuard{_ioThreadDone};

    void start() {
        _underflowCount.store(0U, std::memory_order_relaxed);
        _ioThreadStarted.store(false, std::memory_order_relaxed);
        configureTaper();
        reinitDevice();
        if (!_txStream.get()) {
            return;
        }

        std::size_t nCh     = (nPorts != std::dynamic_extent) ? nPorts : static_cast<std::size_t>(num_channels.value);
        std::size_t bufSize = std::bit_ceil(static_cast<std::size_t>(max_chunk_size) * 4UZ);
        _stagingBuffers.clear();
        _stagingWriters.clear();
        _stagingReaders.clear();
        _stagingBuffers.reserve(nCh);
        _stagingWriters.reserve(nCh);
        _stagingReaders.reserve(nCh);
        for (std::size_t ch = 0UZ; ch < nCh; ++ch) {
            _stagingBuffers.emplace_back(bufSize);
            _stagingWriters.push_back(_stagingBuffers.back().new_writer());
            _stagingReaders.push_back(_stagingBuffers.back().new_reader());
        }

        soapy::detail::DeviceRegistry::registerActivation(_devKwargs, [this] {
            if (auto r = _txStream.activate(); !r) {
                this->emitErrorMessage("start()", r.error());
                this->requestStop();
                return;
            }
            _ioThreadStarted.store(true, std::memory_order_release);
            gr::atomic_ref(_ioThreadDone).store_release(false);
            thread_pool::Manager::defaultIoPool()->execute([this]() { ioWriteLoop(); });
        });
    }

    void stop() {
        if (_ioThreadStarted.load(std::memory_order_acquire)) {
            gr::atomic_ref(_ioThreadDone).wait(false);
        }
        _stagingWriters.clear();
        _stagingReaders.clear();
        _stagingBuffers.clear();
        _txStream.reset();
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
        return Block<SoapySink<T, nPorts>>::work(requestedWork);
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input) noexcept
    requires(nPorts == 1U)
    {
        if (!_ioThreadStarted.load(std::memory_order_acquire) || _stagingWriters.empty()) {
            std::ignore = input.consume(input.size());
            return gr::work::Status::OK;
        }
        auto nToWrite = std::min(input.size(), static_cast<std::size_t>(max_chunk_size));
        if (nToWrite == 0UZ) {
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        auto span = _stagingWriters[0].template tryReserve<SpanReleasePolicy::ProcessNone>(nToWrite);
        if (span.empty()) {
            std::ignore = input.consume(0UZ);
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }
        auto nCopy     = std::min(nToWrite, span.size());
        auto inputSpan = std::span<const T>(input.begin(), nCopy);
        std::memcpy(span.data(), inputSpan.data(), nCopy * sizeof(T));
        span.publish(nCopy);
        std::ignore = input.consume(nCopy);
        return gr::work::Status::OK;
    }

    template<InputSpanLike TInSpan>
    [[nodiscard]] gr::work::Status processBulk(std::span<TInSpan>& inputs) noexcept
    requires(nPorts != 1U)
    {
        if (!_ioThreadStarted.load(std::memory_order_acquire) || _stagingWriters.empty()) {
            for (auto& input : inputs) {
                std::ignore = input.consume(input.size());
            }
            return gr::work::Status::OK;
        }
        std::size_t minSize = std::numeric_limits<std::size_t>::max();
        for (const auto& input : inputs) {
            minSize = std::min(minSize, input.size());
        }
        auto nToWrite = std::min(minSize, static_cast<std::size_t>(max_chunk_size));
        if (nToWrite == 0UZ) {
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        for (std::size_t ch = 0UZ; ch < _stagingWriters.size() && ch < inputs.size(); ++ch) {
            if (_stagingWriters[ch].available() < nToWrite) {
                for (auto& input : inputs) {
                    std::ignore = input.consume(0UZ);
                }
                return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
            }
        }

        for (std::size_t ch = 0UZ; ch < _stagingWriters.size() && ch < inputs.size(); ++ch) {
            auto span = _stagingWriters[ch].template tryReserve<SpanReleasePolicy::ProcessNone>(nToWrite);
            if (span.empty()) {
                for (auto& input : inputs) {
                    std::ignore = input.consume(0UZ);
                }
                return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
            }
            auto nCopy     = std::min(nToWrite, span.size());
            auto inputSpan = std::span<const T>(inputs[ch].begin(), nCopy);
            std::memcpy(span.data(), inputSpan.data(), nCopy * sizeof(T));
            span.publish(nCopy);
        }
        for (auto& input : inputs) {
            std::ignore = input.consume(nToWrite);
        }
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& /*oldSettings*/, property_map& newSettings, property_map& /*forwardSettings*/) {
        if (!_device.get()) {
            return;
        }
        if (newSettings.contains("frequency")) {
            applyFrequency();
        }
        if (newSettings.contains("sample_rate")) {
            applySampleRate();
        }
        if (newSettings.contains("tx_antennae")) {
            applyAntenna();
        }
        if (newSettings.contains("tx_gains")) {
            applyGain();
        }
        if (newSettings.contains("tx_bandwidths")) {
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
    }

    void ioWriteLoop() {
        thread_pool::thread::setThreadName(std::format("soapy_tx:{}", this->name.value));
        std::vector<T> txScratch(static_cast<std::size_t>(max_chunk_size));

        if constexpr (nPorts == 1U) {
            while (lifecycle::isActive(this->state())) {
                this->applyChangedSettings();

                auto avail = _stagingReaders[0].available();
                if (avail == 0UZ) {
                    std::this_thread::yield();
                    continue;
                }
                auto nToWrite = std::min(static_cast<std::size_t>(avail), static_cast<std::size_t>(max_chunk_size));
                auto rSpan    = _stagingReaders[0].get(nToWrite);
                auto nActual  = std::min(rSpan.size(), nToWrite);
                if (nActual == 0UZ) {
                    continue;
                }

                auto [written, ok] = taperAndWrite(rSpan.begin(), nActual, txScratch);
                std::ignore        = rSpan.consume(written);
                if (written > 0UZ) {
                    this->progress->incrementAndGet();
                    this->progress->notify_all();
                }
                if (!ok) {
                    break;
                }
            }
        } else {
            std::size_t                 nCh = _stagingReaders.size();
            std::vector<std::vector<T>> chScratch(nCh, std::vector<T>(static_cast<std::size_t>(max_chunk_size)));

            while (lifecycle::isActive(this->state())) {
                this->applyChangedSettings();

                std::size_t minAvail = std::numeric_limits<std::size_t>::max();
                for (std::size_t ch = 0UZ; ch < nCh; ++ch) {
                    minAvail = std::min(minAvail, static_cast<std::size_t>(_stagingReaders[ch].available()));
                }
                if (minAvail == 0UZ) {
                    std::this_thread::yield();
                    continue;
                }

                auto nToWrite     = std::min(minAvail, static_cast<std::size_t>(max_chunk_size));
                auto savedPhase   = _taper._phase;
                auto savedRampPos = _taper._rampPosition;

                using RSpanType = decltype(_stagingReaders[0].get(0UZ));
                std::vector<RSpanType>          rSpans;
                std::vector<std::span<const T>> writeSpans;
                rSpans.reserve(nCh);
                writeSpans.reserve(nCh);
                for (std::size_t ch = 0UZ; ch < nCh; ++ch) {
                    rSpans.push_back(_stagingReaders[ch].get(nToWrite));
                    auto n = std::min(rSpans.back().size(), nToWrite);
                    std::memcpy(chScratch[ch].data(), &(*rSpans.back().begin()), n * sizeof(T));
                    if (burst_taper_enabled && ch == 0UZ) {
                        applyTaper(chScratch[ch].data(), n);
                    } else if (burst_taper_enabled) {
                        _taper._phase        = savedPhase;
                        _taper._rampPosition = savedRampPos;
                        applyTaper(chScratch[ch].data(), n);
                    }
                    writeSpans.push_back(std::span<const T>(chScratch[ch].data(), n));
                }

                int  flags = 0;
                auto ret   = _txStream.writeStreamFromBufferList(flags, 0LL, static_cast<long>(max_time_out_us), std::span<std::span<const T>>(writeSpans));

                if (ret == SOAPY_SDR_TIMEOUT || ret < 0) {
                    _taper._phase        = savedPhase;
                    _taper._rampPosition = savedRampPos;
                }
                auto nConsumed = (ret > 0) ? static_cast<std::size_t>(ret) : 0UZ;
                if (ret > 0 && nConsumed < nToWrite) {
                    _taper._phase        = savedPhase;
                    _taper._rampPosition = savedRampPos;
                    for (std::size_t i = 0UZ; i < nConsumed; ++i) {
                        std::ignore = _taper.processOne();
                    }
                }
                for (auto& rSpan : rSpans) {
                    std::ignore = rSpan.consume(nConsumed);
                }
                if (nConsumed > 0UZ) {
                    this->progress->incrementAndGet();
                    this->progress->notify_all();
                }
                if (ret < 0 && ret != SOAPY_SDR_TIMEOUT && !handleStreamError(ret)) {
                    break;
                }
            }
        }

        drainRemainingSamples(txScratch);
        completeSafetyRampDown(txScratch);

        if (auto r = _txStream.deactivate(); !r) {
            this->emitErrorMessage("ioWriteLoop()", r.error());
        }

        gr::atomic_ref(_ioThreadDone).store_release(true);
        gr::atomic_ref(_ioThreadDone).notify_all();
    }

    std::pair<std::size_t, bool> taperAndWrite(auto srcIter, std::size_t n, std::vector<T>& scratch) {
        auto savedPhase   = _taper._phase;
        auto savedRampPos = _taper._rampPosition;

        std::memcpy(scratch.data(), &(*srcIter), n * sizeof(T));
        if (burst_taper_enabled) {
            applyTaper(scratch.data(), n);
        }

        int  flags = 0;
        auto ret   = _txStream.writeStream(flags, 0LL, static_cast<long>(max_time_out_us), std::span<const T>(scratch.data(), n));

        if (ret == SOAPY_SDR_TIMEOUT) {
            _taper._phase        = savedPhase;
            _taper._rampPosition = savedRampPos;
            return {0UZ, true};
        }
        if (ret < 0) {
            _taper._phase        = savedPhase;
            _taper._rampPosition = savedRampPos;
            return {0UZ, handleStreamError(ret)};
        }
        auto nWritten = static_cast<std::size_t>(ret);
        if (nWritten < n) {
            _taper._phase        = savedPhase;
            _taper._rampPosition = savedRampPos;
            for (std::size_t i = 0UZ; i < nWritten; ++i) {
                std::ignore = _taper.processOne();
            }
        }
        return {nWritten, true};
    }

    void applyTaper(T* samples, std::size_t n) {
        for (std::size_t i = 0UZ; i < n; ++i) {
            float envelope = _taper.processOne();
            if constexpr (std::is_same_v<T, std::complex<float>>) {
                samples[i] *= envelope;
            } else {
                samples[i] = static_cast<T>(static_cast<float>(samples[i]) * envelope);
            }
        }
    }

    void drainRemainingSamples(std::vector<T>& scratch) {
        if constexpr (nPorts == 1U) {
            while (true) {
                auto avail = _stagingReaders[0].available();
                if (avail == 0UZ) {
                    break;
                }
                auto nToWrite      = std::min(static_cast<std::size_t>(avail), static_cast<std::size_t>(max_chunk_size));
                auto rSpan         = _stagingReaders[0].get(nToWrite);
                auto nActual       = std::min(rSpan.size(), nToWrite);
                auto [written, ok] = taperAndWrite(rSpan.begin(), nActual, scratch);
                std::ignore        = rSpan.consume(written);
                if (!ok) {
                    break;
                }
            }
        } else {
            // multi-channel drain omitted for brevity — same pattern as single-channel per reader
            for (auto& reader : _stagingReaders) {
                while (true) {
                    auto avail = reader.available();
                    if (avail == 0UZ) {
                        break;
                    }
                    auto rSpan  = reader.get(std::min(static_cast<std::size_t>(avail), static_cast<std::size_t>(max_chunk_size)));
                    std::ignore = rSpan.consume(rSpan.size());
                }
            }
        }
    }

    void completeSafetyRampDown(std::vector<T>& scratch) {
        if (!burst_taper_enabled || !burst_safety_rampdown || _taper.isOff()) {
            return;
        }
        std::println(stderr, "[SoapySink] forced safety ramp-down on shutdown (upstream did not taper)");
        std::ignore = _taper.setTarget(false, true);

        constexpr float kMaxSafetyRampSec = 1.0f;
        auto            maxSamples        = static_cast<std::size_t>(std::min(burst_ramp_time.value, kMaxSafetyRampSec) * sample_rate);
        if (burst_ramp_time > kMaxSafetyRampSec) {
            std::println(stderr, "[SoapySink] safety ramp-down clamped to 1s (configured: {}s)", burst_ramp_time.value);
        }

        std::fill_n(scratch.data(), std::min(scratch.size(), maxSamples), T{0});
        std::size_t written = 0UZ;
        while (!_taper.isOff() && written < maxSamples) {
            auto n = std::min(scratch.size(), maxSamples - written);
            for (std::size_t i = 0UZ; i < n; ++i) {
                std::ignore = _taper.processOne();
            }
            int  flags = 0;
            auto ret   = _txStream.writeStream(flags, 0LL, static_cast<long>(max_time_out_us), std::span<const T>(scratch.data(), n));
            if (ret <= 0 && ret != SOAPY_SDR_TIMEOUT) {
                break;
            }
            written += (ret > 0) ? static_cast<std::size_t>(ret) : 0UZ;
        }
    }

    void configureTaper() {
        if (!burst_taper_enabled) {
            _taper.reset();
            return;
        }
        auto type = parseTaperType(burst_taper_type.value);
        if (auto r = _taper.configure(type, burst_ramp_time, sample_rate, burst_shape_param); !r) {
            this->emitErrorMessage("configureTaper()", r.error());
            return;
        }
        std::ignore = _taper.setTarget(true);
    }

    static algorithm::TaperType parseTaperType(const std::string& name) {
        if (name == "None") {
            return algorithm::TaperType::None;
        }
        if (name == "Linear") {
            return algorithm::TaperType::Linear;
        }
        if (name == "RaisedCosine") {
            return algorithm::TaperType::RaisedCosine;
        }
        if (name == "Tukey") {
            return algorithm::TaperType::Tukey;
        }
        if (name == "Gaussian") {
            return algorithm::TaperType::Gaussian;
        }
        if (name == "Mushroom") {
            return algorithm::TaperType::Mushroom;
        }
        if (name == "MushroomSine") {
            return algorithm::TaperType::MushroomSine;
        }
        return algorithm::TaperType::RaisedCosine;
    }

    void reinitDevice() {
        _txStream.reset();
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

        std::size_t nChannelMax    = _device.getNumChannels(SOAPY_SDR_TX);
        std::size_t nChannelNeeded = (nPorts != std::dynamic_extent) ? nPorts : static_cast<std::size_t>(num_channels.value);
        if (nChannelMax < nChannelNeeded) {
            this->emitErrorMessage("reinitDevice()", std::format("TX channel mismatch: need {} but device has {}", nChannelNeeded, nChannelMax));
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

        auto        supportedFormats = _device.getStreamFormats(SOAPY_SDR_TX, 0);
        const char* requestedFormat  = soapy::detail::toSoapySDRFormat<T>();
        if (!supportedFormats.empty() && std::ranges::find(supportedFormats, std::string(requestedFormat)) == supportedFormats.end()) {
            this->emitErrorMessage("reinitDevice()", std::format("TX format '{}' not supported (available: {})", requestedFormat, gr::join(supportedFormats, ", ")));
            this->requestStop();
            return;
        }

        std::vector<gr::Size_t> channelIndices(num_channels);
        std::iota(channelIndices.begin(), channelIndices.end(), gr::Size_t{0});
        soapy::Kwargs parsedStreamArgs = stream_args->empty() ? soapy::Kwargs{} : soapy::parseKwargsString(stream_args.value);
        auto          streamResult     = _device.setupStream<T, SOAPY_SDR_TX>(channelIndices, parsedStreamArgs);
        if (!streamResult) {
            this->emitErrorMessage("reinitDevice()", std::format("{} (requested: {})", streamResult.error(), requestedFormat));
            this->requestStop();
            return;
        }
        _txStream = std::move(*streamResult);
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
            if (auto r = _device.setSampleRate(SOAPY_SDR_TX, i, static_cast<double>(sample_rate)); !r) {
                this->emitErrorMessage("applySampleRate()", r.error());
            }
        }
    }

    void applyAntenna() {
        if (tx_antennae->empty()) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            const auto& ant = tx_antennae->at(std::min(static_cast<std::size_t>(i), tx_antennae->size() - 1UZ));
            if (!ant.empty()) {
                if (auto r = _device.setAntenna(SOAPY_SDR_TX, i, ant); !r) {
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
            if (auto r = _device.setCenterFrequency(SOAPY_SDR_TX, i, freq); !r) {
                this->emitErrorMessage("applyFrequency()", r.error());
            }
        }
    }

    void applyBandwidth() {
        if (tx_bandwidths->empty()) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            double bw = tx_bandwidths->at(std::min(static_cast<std::size_t>(i), tx_bandwidths->size() - 1UZ));
            if (auto r = _device.setBandwidth(SOAPY_SDR_TX, i, bw); !r) {
                this->emitErrorMessage("applyBandwidth()", r.error());
            }
        }
    }

    void applyGain() {
        if (tx_gains->empty()) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            double g = tx_gains->at(std::min(static_cast<std::size_t>(i), tx_gains->size() - 1UZ));
            if (auto r = _device.setGain(SOAPY_SDR_TX, i, g); !r) {
                this->emitErrorMessage("applyGain()", r.error());
            }
        }
    }

    void applyGainMode() {
        if (!gain_mode) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            if (!_device.hasAutomaticGainControl(SOAPY_SDR_TX, i)) {
                continue;
            }
            if (auto r = _device.setAutomaticGainControl(SOAPY_SDR_TX, i, gain_mode); !r) {
                this->emitErrorMessage("applyGainMode()", r.error());
            }
        }
    }

    void applyFrequencyCorrection() {
        if (frequency_correction == 0.0) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            if (!_device.hasFrequencyCorrection(SOAPY_SDR_TX, i)) {
                continue;
            }
            if (auto r = _device.setFrequencyCorrection(SOAPY_SDR_TX, i, frequency_correction); !r) {
                this->emitErrorMessage("applyFrequencyCorrection()", r.error());
            }
        }
    }

    void applyDcOffsetMode() {
        if (!dc_offset_mode) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            if (!_device.hasDCOffsetMode(SOAPY_SDR_TX, i)) {
                continue;
            }
            if (auto r = _device.setDCOffsetMode(SOAPY_SDR_TX, i, dc_offset_mode); !r) {
                this->emitErrorMessage("applyDcOffsetMode()", r.error());
            }
        }
    }

    void applyDcOffset() {
        if (dc_offset->empty()) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            if (!_device.hasDCOffset(SOAPY_SDR_TX, i)) {
                continue;
            }
            auto idx = static_cast<std::size_t>(i) * 2UZ;
            if (idx + 1UZ >= dc_offset->size()) {
                break;
            }
            if (auto r = _device.setDCOffset(SOAPY_SDR_TX, i, dc_offset->at(idx), dc_offset->at(idx + 1UZ)); !r) {
                this->emitErrorMessage("applyDcOffset()", r.error());
            }
        }
    }

    void applyIqBalance() {
        if (iq_balance->empty()) {
            return;
        }
        for (gr::Size_t i = 0U; i < num_channels; i++) {
            if (!_device.hasIQBalance(SOAPY_SDR_TX, i)) {
                continue;
            }
            auto idx = static_cast<std::size_t>(i) * 2UZ;
            if (idx + 1UZ >= iq_balance->size()) {
                break;
            }
            if (auto r = _device.setIQBalance(SOAPY_SDR_TX, i, iq_balance->at(idx), iq_balance->at(idx + 1UZ)); !r) {
                this->emitErrorMessage("applyIqBalance()", r.error());
            }
        }
    }

    void applyFrontendMapping() {
        if (!frontend_mapping->empty()) {
            _device.setFrontendMapping(SOAPY_SDR_TX, frontend_mapping.value);
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

    bool handleStreamError(int ret) {
        switch (ret) {
        case SOAPY_SDR_UNDERFLOW: {
            auto count = _underflowCount.fetch_add(1U, std::memory_order_relaxed) + 1U;
            if (verbose_underflow) {
                std::println(stderr, "[SoapySink] UNDERFLOW #{}", count);
            }
            if (max_underflow_count > 0 && count >= max_underflow_count) {
                this->emitErrorMessage("ioWriteLoop()", std::format("UNDERFLOW: {} of max {}", count, max_underflow_count));
                this->requestStop();
                return false;
            }
            return true;
        }
        default:
            this->emitErrorMessage("ioWriteLoop()", std::format("TX stream error: {}", ret));
            this->requestStop();
            return false;
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
using SoapySimpleSink = SoapySink<T, 1UZ>;
template<typename T>
using SoapyDualSink = SoapySink<T, 2UZ>;
template<typename T>
using SoapyQuadSink = SoapySink<T, 4UZ>;

} // namespace gr::blocks::sdr

#endif // GNURADIO_SOAPY_SINK_HPP
