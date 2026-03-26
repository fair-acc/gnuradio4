#ifndef GNURADIO_SOAPY_SOURCE_HPP
#define GNURADIO_SOAPY_SOURCE_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <gnuradio-4.0/sdr/SoapyRaiiWrapper.hpp>

namespace gr::blocks::sdr {

namespace detail {
inline bool equalWithinOnePercent(const auto& a, const auto& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](double x, double y) { return std::abs(x - y) <= 0.01 * std::max(std::abs(x), std::abs(y)); });
}
} // namespace detail

GR_REGISTER_BLOCK("gr::blocks::sdr::SoapySource", gr::blocks::sdr::SoapySource, ([T], 1UZ), [ uint8_t, int16_t, std::complex<float> ])
GR_REGISTER_BLOCK("gr::blocks::sdr::SoapyDualSource", gr::blocks::sdr::SoapySource, ([T], 2UZ), [ uint8_t, int16_t, std::complex<float> ])

template<typename T, std::size_t nPorts = std::dynamic_extent>
struct SoapySource : Block<SoapySource<T, nPorts>> {
    using Description = Doc<R"(SoapySDR source block for SDR hardware.
Supports single and multi-channel RX via SoapySDR's device-agnostic API.
Uses a dedicated IO thread to decouple USB/hardware latency from the scheduler.
Tested with RTL-SDR and LimeSDR drivers.)">;

    using TSizeChecker = Limits<0U, std::numeric_limits<std::uint32_t>::max(), [](std::uint32_t x) { return std::has_single_bit(x); }>;
    using TBasePort    = PortOut<T>;
    using TPortType    = std::conditional_t<nPorts == 1U, TBasePort, std::conditional_t<nPorts == std::dynamic_extent, std::vector<TBasePort>, std::array<TBasePort, nPorts>>>;

    TPortType out;

    // SigMF-aligned tags: core:sample_rate (global), core:frequency (captures), core:num_channels (global)
    Annotated<std::string, "device", Visible, Doc<"SoapySDR driver name">>                                                       device;
    Annotated<std::string, "device_parameter", Visible, Doc<"additional driver parameters">>                                     device_parameter;
    Annotated<float, "sample_rate", Unit<"Hz">, Visible, Doc<"ADC sample rate (SigMF core:sample_rate)">>                        sample_rate  = 1'000'000.f;
    Annotated<gr::Size_t, "num_channels", Visible, Doc<"number of RX channels (SigMF core:num_channels)">>                       num_channels = 1U;
    Annotated<std::vector<std::string>, "rx_antennae", Visible, Doc<"per-channel RX antenna selection">>                         rx_antennae;
    Annotated<std::vector<double>, "frequency", Unit<"Hz">, Visible, Doc<"per-channel center frequency (SigMF core:frequency)">> frequency     = initDefaultValues(107'000'000.);
    Annotated<std::vector<double>, "rx_bandwidths", Unit<"Hz">, Visible, Doc<"per-channel RX RF bandwidth">>                     rx_bandwidths = initDefaultValues(500'000.);
    Annotated<std::vector<double>, "rx_gains", Unit<"dB">, Visible, Doc<"per-channel RX tuner gain">>                            rx_gains      = initDefaultValues(10.);

    Annotated<std::uint32_t, "max_chunk_size", Doc<"max samples per read (ideally N x 512)">, Visible, TSizeChecker> max_chunk_size     = 512U << 4U;
    Annotated<std::uint32_t, "max_time_out_us", Unit<"us">, Doc<"SoapySDR polling timeout">>                         max_time_out_us    = 1'000;
    Annotated<gr::Size_t, "max_overflow_count", Doc<"max consecutive overflows before stop (0 = disable)">>          max_overflow_count = 10U;
    Annotated<gr::Size_t, "max_fragment_count", Doc<"max consecutive fragments before stop (0 = disable)">>          max_fragment_count = 100U;
    Annotated<bool, "verbose_overflow", Doc<"log each overflow event">>                                              verbose_overflow   = false;

    GR_MAKE_REFLECTABLE(SoapySource, out, device, device_parameter, sample_rate, num_channels, rx_antennae, frequency, rx_bandwidths, rx_gains, max_chunk_size, max_time_out_us, max_overflow_count, max_fragment_count, verbose_overflow);

    soapy::Device                          _device{};
    soapy::Device::Stream<T, SOAPY_SDR_RX> _rxStream{};
    bool                                   _ioThreadDone = true;
    std::atomic<gr::Size_t>                _overFlowCount{0U};
    gr::Size_t                             _fragmentCount = 0U;

    struct IoThreadGuard {
        bool& done;
        ~IoThreadGuard() { gr::atomic_ref(done).wait(false); }
    };
    IoThreadGuard _ioGuard{_ioThreadDone};

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (!_device.get()) {
            return;
        }

        if (newSettings.contains("rx_antennae")) {
            applyAntenna();
        }
        if (newSettings.contains("frequency") || newSettings.contains("sample_rate")) {
            applyFrequency();
        }
        if (newSettings.contains("rx_gains")) {
            applyGain();
        }
    }

    void start() {
        _overFlowCount.store(0U, std::memory_order_relaxed);
        _fragmentCount = 0U;
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

                auto span = outWriter.template tryReserve<SpanReleasePolicy::ProcessNone>(nSamples);
                if (span.empty()) {
                    continue;
                }
                std::memcpy(span.data(), readBuf.data(), nSamples * sizeof(T));
                span.publish(nSamples);

                this->progress->incrementAndGet();
                this->progress->notify_all();
            }
        } else {
            std::vector<std::vector<T>> readBufs(nCh, std::vector<T>(kReadSize));
            // cache per-channel writer references
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

                bool allAvailable = std::ranges::all_of(outWriters, [nSamples](auto& w) { return w.get().available() >= nSamples; });
                if (!allAvailable) {
                    continue; // backpressure on any channel — retry entire read
                }
                for (std::size_t ch = 0UZ; ch < outWriters.size(); ++ch) {
                    auto span = outWriters[ch].get().template tryReserve<SpanReleasePolicy::ProcessNone>(nSamples);
                    std::memcpy(span.data(), readBufs[ch].data(), nSamples * sizeof(T));
                    span.publish(nSamples);
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

    void reinitDevice() {
        _rxStream.reset();
        auto devResult = soapy::Device::make(soapy::Kwargs{{"driver", device.value}});
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
            this->emitErrorMessage("reinitDevice()", std::format("format '{}' not supported by device (available: {})", requestedFormat, gr::join(supportedFormats, ", ")));
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
