#ifndef GNURADIO_LOOPBACK_DEVICE_HPP
#define GNURADIO_LOOPBACK_DEVICE_HPP

#include <algorithm>
#include <atomic>
#include <cassert>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <gnuradio-4.0/CircularBuffer.hpp>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-W#warnings"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#endif
#include <SoapySDR/Device.hpp>
#include <SoapySDR/Formats.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <gnuradio-4.0/algorithm/rng/GaussianNoise.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

namespace gr::blocks::sdr::loopback {

using CF32 = std::complex<float>;

enum class DeviceMode { Loopback, RxOnly, TxOnly };

/**
 * @brief SoapySDR-compatible loopback device with pluggable per-channel model.
 *
 * Provides a virtual SDR device for CI testing and channel simulation without
 * physical hardware. Three operating modes:
 *  - loopback (default): TX samples routed through ChannelModel to RX side
 *  - rxOnly: internal tone generator → ChannelModel → RX (no TX needed)
 *  - txOnly: TX accepted and discarded (null sink, no RX data produced)
 *
 * Features:
 *  - instance registry: driver=loopback#N for RX/TX pairing and parallel instances
 *  - configurable N channels with per-channel models (default: passthrough)
 *  - sample format conversion: CF32 <-> CS16 <-> CU8
 *  - optional rate-limited readStream (simulate_timing, default on for rxOnly)
 *  - pluggable channel model via setChannelModel() or Soapy writeSetting()
 *  - built-in models: passthrough, attenuation, AWGN, delay, composable chain
 *
 * Channel models reuse GR4 algorithms (gr::rng::GaussianNoise, etc.) called
 * directly without a scheduler. The std::function interface is extensible to
 * full sub-graphs (manual stepping) or external simulators (digital twin).
 *
 * Instance registry (in SoapyLoopbackModule.cpp):
 *  - driver=loopback     -> find first existing instance, or create #0
 *  - driver=loopback#0   -> find-or-create instance 0
 *  - driver=loopback#3   -> find-or-create instance 3
 *  Both RX (SoapySource) and TX (SoapySink) blocks resolve to the same
 *  LoopbackDevice when using the same instance ID, enabling TX->RX loopback.
 *
 * Threading model: single TX thread + single RX thread. Stream active flags
 * are atomic for safe concurrent activation/deactivation.
 *
 * Build targets:
 *  - gr-sdr-loopback-static: linked into test binaries for direct C++ access
 *  - gr-sdr-loopback.so: SoapySDR module for use in regular GR4 flowgraphs
 *
 * clean rewrite, but inspired by https://github.com/JuliaTelecom/SoapyLoopback
 */

struct ChannelModel {
    using ProcessFn = std::function<void(std::span<const CF32> in, std::span<CF32> out)>;

    ProcessFn process;

    static ChannelModel passthrough() {
        return {[](std::span<const CF32> in, std::span<CF32> out) { std::ranges::copy_n(in.begin(), static_cast<std::ptrdiff_t>(std::min(in.size(), out.size())), out.begin()); }};
    }

    static ChannelModel attenuation(float dB) {
        float scale = std::pow(10.f, dB / 20.f);
        return {[scale](std::span<const CF32> in, std::span<CF32> out) {
            auto n = std::min(in.size(), out.size());
            std::ranges::transform(in.first(n), out.begin(), [scale](CF32 s) { return s * scale; });
        }};
    }

    static ChannelModel awgn(float noiseFloorDbfs) {
        float noiseAmplitude = std::pow(10.f, noiseFloorDbfs / 20.f);
        auto  rng            = std::make_shared<gr::rng::Xoshiro256pp>();
        auto  noise          = std::make_shared<gr::rng::GaussianNoise<float>>(*rng);
        return {[noiseAmplitude, rng, noise](std::span<const CF32> in, std::span<CF32> out) {
            auto n = std::min(in.size(), out.size());
            std::ranges::transform(in.first(n), out.begin(), [&](CF32 s) { return s + noiseAmplitude * noise->complexSample(); });
        }};
    }

    static ChannelModel delay(std::size_t nSamples) {
        if (nSamples == 0UZ) {
            return passthrough();
        }
        struct DelayState {
            std::vector<CF32> line;
            std::size_t       pos = 0UZ;
        };
        auto state = std::make_shared<DelayState>(DelayState{std::vector<CF32>(nSamples, CF32{0.f, 0.f})});
        return {[state, nSamples](std::span<const CF32> in, std::span<CF32> out) {
            auto  n    = std::min(in.size(), out.size());
            auto* line = state->line.data();
            auto& pos  = state->pos;
            for (std::size_t i = 0UZ; i < n; ++i) {
                auto sample = in[i];
                out[i]      = line[pos];
                line[pos]   = sample;
                pos         = (pos + 1UZ) % nSamples;
            }
        }};
    }

    static ChannelModel chain(std::initializer_list<ChannelModel> stages) { return chain(std::vector<ChannelModel>(stages)); }

    static ChannelModel chain(std::vector<ChannelModel> stages) {
        auto models = std::make_shared<std::vector<ChannelModel>>(std::move(stages));
        return {[models](std::span<const CF32> in, std::span<CF32> out) {
            auto n = std::min(in.size(), out.size());
            if (models->empty()) {
                std::ranges::copy_n(in.begin(), static_cast<std::ptrdiff_t>(n), out.begin());
                return;
            }
            (*models)[0].process(in, out);
            for (std::size_t s = 1UZ; s < models->size(); ++s) {
                (*models)[s].process(out.subspan(0, n), out.subspan(0, n));
            }
        }};
    }
};

class LoopbackDevice; // forward declaration for DeviceRegistry

struct DeviceRegistry {
    static std::mutex& mutex() {
        static auto* m = new std::mutex; // intentional leak — must outlive all SoapySDR module teardown
        return *m;
    }
    static std::map<std::size_t, std::weak_ptr<LoopbackDevice>>& instances() {
        static auto* map = new std::map<std::size_t, std::weak_ptr<LoopbackDevice>>; // intentional leak
        return *map;
    }

    static std::size_t parseInstanceId(const SoapySDR::Kwargs& args) {
        auto it = args.find("driver");
        if (it == args.end()) {
            return 0UZ;
        }
        const auto& driver = it->second;
        if (auto hashPos = driver.find('#'); hashPos != std::string::npos) {
            std::size_t id  = 0UZ;
            auto        sub = driver.substr(hashPos + 1);
            std::from_chars(sub.data(), sub.data() + sub.size(), id);
            return id;
        }
        return 0UZ;
    }

    static bool isLoopbackDriver(const SoapySDR::Kwargs& args) {
        auto it = args.find("driver");
        if (it == args.end()) {
            return false;
        }
        const auto& driver = it->second;
        return driver == "loopback" || driver.starts_with("loopback#");
    }

    // defined after LoopbackDevice (below)
    static std::shared_ptr<LoopbackDevice> findOrCreate(std::size_t instanceId, const SoapySDR::Kwargs& args = {});
};

class LoopbackDevice : public SoapySDR::Device {
    static constexpr std::size_t kDefaultBufferSize = 65536UZ;

    using RxBuffer = gr::CircularBuffer<CF32>;
    using RxWriter = decltype(std::declval<RxBuffer>().new_writer());
    using RxReader = decltype(std::declval<RxBuffer>().new_reader());

    enum class SampleFormat { cf32, cs16, cu8 };

    // per-instance sentinel addresses for stream handles (valid pointers, never dereferenced)
    char _rxStreamSentinel = 0;
    char _txStreamSentinel = 0;

    struct ChannelState {
        double               frequency     = 100e6;
        double               sampleRate    = 1e6;
        double               bandwidth     = 0.0;
        double               gain          = 0.0;
        double               ppmCorrection = 0.0;
        bool                 dcOffsetMode  = false;
        std::complex<double> dcOffset{0.0, 0.0};
        std::complex<double> iqBalance{1.0, 0.0};
        std::string          antenna = "RX";
        ChannelModel         model   = ChannelModel::passthrough();
        double               rxPhase = 0.0; // phase accumulator for rxOnly tone generator
        RxBuffer             rxBuffer;
        RxWriter             rxWriter;
        RxReader             rxReader;
        std::vector<CF32>    txScratch; // pre-allocated for non-CF32 TX format conversion

        explicit ChannelState(std::size_t bufferSize) : rxBuffer(bufferSize), rxWriter(rxBuffer.new_writer()), rxReader(rxBuffer.new_reader()) {}
    };

    std::size_t                                          _instanceId  = 0UZ;
    std::size_t                                          _numChannels = 1UZ;
    std::size_t                                          _bufferSize  = kDefaultBufferSize;
    DeviceMode                                           _deviceMode  = DeviceMode::Loopback;
    std::atomic<bool>                                    _simulateTiming{false};
    std::vector<CF32>                                    _rxToneScratch;  // reusable per-readStream call
    std::vector<CF32>                                    _rxModelScratch; // reusable per-readStream call
    std::vector<std::unique_ptr<ChannelState>>           _channels;
    std::atomic<bool>                                    _rxStreamActive{false};
    std::atomic<bool>                                    _txStreamActive{false};
    SampleFormat                                         _rxFormatEnum = SampleFormat::cf32;
    SampleFormat                                         _txFormatEnum = SampleFormat::cf32;
    std::vector<std::size_t>                             _rxChannels;
    std::vector<std::size_t>                             _txChannels;
    std::chrono::steady_clock::time_point                _lastReadTime;
    std::string                                          _rxFrontendMapping;
    std::string                                          _txFrontendMapping;
    double                                               _masterClockRate    = 0.0;
    double                                               _referenceClockRate = 0.0;
    std::string                                          _clockSource        = "internal";
    std::map<std::string, unsigned>                      _gpioValues;
    std::map<std::string, unsigned>                      _gpioDirValues;
    std::map<std::pair<std::string, unsigned>, unsigned> _registers;

public:
    explicit LoopbackDevice(const SoapySDR::Kwargs& args) {
        _instanceId = DeviceRegistry::parseInstanceId(args);
        if (auto it = args.find("num_channels"); it != args.end()) {
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), _numChannels);
        }
        if (auto it = args.find("buffer_size"); it != args.end()) {
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), _bufferSize);
        }
        if (auto it = args.find("device_mode"); it != args.end()) {
            _deviceMode = parseDeviceMode(it->second);
        }
        if (_deviceMode == DeviceMode::RxOnly) {
            _simulateTiming.store(true, std::memory_order_relaxed);
        }
        _channels.reserve(_numChannels);
        for (std::size_t i = 0UZ; i < _numChannels; ++i) {
            _channels.push_back(std::make_unique<ChannelState>(_bufferSize));
        }
    }

    [[nodiscard]] std::size_t instanceId() const { return _instanceId; }

    void setChannelModel(ChannelModel model) {
        assert(!_txStreamActive.load(std::memory_order_relaxed) && "setChannelModel must be called before activateStream(TX)");
        if (!model.process) {
            model = ChannelModel::passthrough();
        }
        for (auto& ch : _channels) {
            ch->model = model;
        }
    }

    void setChannelModel(std::size_t channel, ChannelModel model) {
        assert(!_txStreamActive.load(std::memory_order_relaxed) && "setChannelModel must be called before activateStream(TX)");
        if (!model.process) {
            model = ChannelModel::passthrough();
        }
        if (channel < _numChannels) {
            _channels[channel]->model = std::move(model);
        }
    }

    std::string      getDriverKey() const override { return "loopback"; }
    std::string      getHardwareKey() const override { return "loopback"; }
    SoapySDR::Kwargs getHardwareInfo() const override { return {{"driver", "loopback"}, {"instance_id", std::to_string(_instanceId)}, {"version", "1.0"}}; }

    void             setFrontendMapping(const int direction, const std::string& mapping) override { (direction == SOAPY_SDR_RX) ? _rxFrontendMapping = mapping : _txFrontendMapping = mapping; }
    std::string      getFrontendMapping(const int direction) const override { return (direction == SOAPY_SDR_RX) ? _rxFrontendMapping : _txFrontendMapping; }
    size_t           getNumChannels(const int /*direction*/) const override { return _numChannels; }
    SoapySDR::Kwargs getChannelInfo(const int /*direction*/, const size_t /*channel*/) const override { return {}; }
    bool             getFullDuplex(const int /*direction*/, const size_t /*channel*/) const override { return true; }

    std::vector<std::string> listAntennas(const int direction, const size_t /*channel*/) const override { return (direction == SOAPY_SDR_RX) ? std::vector<std::string>{"RX"} : std::vector<std::string>{"TX"}; }
    void                     setAntenna(const int /*direction*/, const size_t channel, const std::string& name) override {
        if (channel < _numChannels) {
            _channels[channel]->antenna = name;
        }
    }
    std::string getAntenna(const int /*direction*/, const size_t channel) const override { return (channel < _numChannels) ? _channels[channel]->antenna : ""; }

    bool                     hasGainMode(const int /*direction*/, const size_t /*channel*/) const override { return true; }
    void                     setGainMode(const int /*direction*/, const size_t /*channel*/, const bool /*automatic*/) override {}
    bool                     getGainMode(const int /*direction*/, const size_t /*channel*/) const override { return false; }
    std::vector<std::string> listGains(const int /*direction*/, const size_t /*channel*/) const override { return {"TUNER"}; }
    void                     setGain(const int /*direction*/, const size_t channel, const double value) override {
        if (channel < _numChannels) {
            _channels[channel]->gain = value;
        }
    }
    double          getGain(const int /*direction*/, const size_t channel) const override { return (channel < _numChannels) ? _channels[channel]->gain : 0.0; }
    void            setGain(const int direction, const size_t channel, const std::string& /*name*/, const double value) override { setGain(direction, channel, value); }
    double          getGain(const int direction, const size_t channel, const std::string& /*name*/) const override { return getGain(direction, channel); }
    SoapySDR::Range getGainRange(const int /*direction*/, const size_t /*channel*/) const override { return SoapySDR::Range(0.0, 60.0); }
    SoapySDR::Range getGainRange(const int direction, const size_t channel, const std::string& /*name*/) const override { return getGainRange(direction, channel); }

    void setFrequency(const int /*direction*/, const size_t channel, const double frequency, const SoapySDR::Kwargs& /*args*/ = {}) override {
        if (channel < _numChannels) {
            _channels[channel]->frequency = frequency;
        }
    }
    double                   getFrequency(const int /*direction*/, const size_t channel) const override { return (channel < _numChannels) ? _channels[channel]->frequency : 0.0; }
    std::vector<std::string> listFrequencies(const int /*direction*/, const size_t /*channel*/) const override { return {"RF"}; }
    SoapySDR::RangeList      getFrequencyRange(const int /*direction*/, const size_t /*channel*/) const override { return {SoapySDR::Range(1e6, 6e9)}; }
    void                     setFrequency(const int direction, const size_t channel, const std::string& /*name*/, const double frequency, const SoapySDR::Kwargs& args = {}) override { setFrequency(direction, channel, frequency, args); }
    double                   getFrequency(const int direction, const size_t channel, const std::string& /*name*/) const override { return getFrequency(direction, channel); }
    SoapySDR::RangeList      getFrequencyRange(const int direction, const size_t channel, const std::string& /*name*/) const override { return getFrequencyRange(direction, channel); }
    SoapySDR::ArgInfoList    getFrequencyArgsInfo(const int /*direction*/, const size_t /*channel*/) const override { return {}; }

    void setSampleRate(const int /*direction*/, const size_t channel, const double rate) override {
        if (channel < _numChannels) {
            _channels[channel]->sampleRate = rate;
        }
    }
    double              getSampleRate(const int /*direction*/, const size_t channel) const override { return (channel < _numChannels) ? _channels[channel]->sampleRate : 0.0; }
    std::vector<double> listSampleRates(const int /*direction*/, const size_t /*channel*/) const override { return {250e3, 500e3, 1e6, 2e6, 2.048e6, 3.2e6, 10e6, 20e6}; }
    SoapySDR::RangeList getSampleRateRange(const int /*direction*/, const size_t /*channel*/) const override { return {SoapySDR::Range(250e3, 20e6)}; }

    void setBandwidth(const int /*direction*/, const size_t channel, const double bw) override {
        if (channel < _numChannels) {
            _channels[channel]->bandwidth = bw;
        }
    }
    double              getBandwidth(const int /*direction*/, const size_t channel) const override { return (channel < _numChannels) ? _channels[channel]->bandwidth : 0.0; }
    std::vector<double> listBandwidths(const int /*direction*/, const size_t /*channel*/) const override { return {200e3, 500e3, 1e6, 5e6, 10e6, 20e6}; }
    SoapySDR::RangeList getBandwidthRange(const int /*direction*/, const size_t /*channel*/) const override { return {SoapySDR::Range(200e3, 20e6)}; }

    bool hasDCOffsetMode(const int /*direction*/, const size_t /*channel*/) const override { return true; }
    void setDCOffsetMode(const int /*direction*/, const size_t channel, const bool automatic) override {
        if (channel < _numChannels) {
            _channels[channel]->dcOffsetMode = automatic;
        }
    }
    bool getDCOffsetMode(const int /*direction*/, const size_t channel) const override { return (channel < _numChannels) && _channels[channel]->dcOffsetMode; }
    bool hasDCOffset(const int /*direction*/, const size_t /*channel*/) const override { return true; }
    void setDCOffset(const int /*direction*/, const size_t channel, const std::complex<double>& offset) override {
        if (channel < _numChannels) {
            _channels[channel]->dcOffset = offset;
        }
    }
    std::complex<double> getDCOffset(const int /*direction*/, const size_t channel) const override { return (channel < _numChannels) ? _channels[channel]->dcOffset : std::complex<double>{0.0, 0.0}; }
    bool                 hasIQBalance(const int /*direction*/, const size_t /*channel*/) const override { return true; }
    void                 setIQBalance(const int /*direction*/, const size_t channel, const std::complex<double>& balance) override {
        if (channel < _numChannels) {
            _channels[channel]->iqBalance = balance;
        }
    }
    std::complex<double> getIQBalance(const int /*direction*/, const size_t channel) const override { return (channel < _numChannels) ? _channels[channel]->iqBalance : std::complex<double>{1.0, 0.0}; }
    bool                 hasIQBalanceMode(const int /*direction*/, const size_t /*channel*/) const override { return false; }
    bool                 hasFrequencyCorrection(const int /*direction*/, const size_t /*channel*/) const override { return true; }
    void                 setFrequencyCorrection(const int /*direction*/, const size_t channel, const double value) override {
        if (channel < _numChannels) {
            _channels[channel]->ppmCorrection = value;
        }
    }
    double getFrequencyCorrection(const int /*direction*/, const size_t channel) const override { return (channel < _numChannels) ? _channels[channel]->ppmCorrection : 0.0; }

    std::vector<std::string> getStreamFormats(const int /*direction*/, const size_t /*channel*/) const override { return {SOAPY_SDR_CF32, SOAPY_SDR_CS16, SOAPY_SDR_CU8}; }
    std::string              getNativeStreamFormat(const int /*direction*/, const size_t /*channel*/, double& fullScale) const override {
        fullScale = 1.0;
        return SOAPY_SDR_CF32;
    }
    SoapySDR::ArgInfoList getStreamArgsInfo(const int /*direction*/, const size_t /*channel*/) const override { return {}; }
    size_t                getStreamMTU(SoapySDR::Stream* /*stream*/) const override { return _bufferSize; }

    SoapySDR::Stream* setupStream(const int direction, const std::string& format, const std::vector<size_t>& channels = {}, const SoapySDR::Kwargs& /*args*/ = {}) override {
        if (direction == SOAPY_SDR_RX) {
            _rxFormatEnum = parseSampleFormat(format);
            _rxChannels   = channels.empty() ? std::vector<size_t>{0} : channels;
            return reinterpret_cast<SoapySDR::Stream*>(&_rxStreamSentinel);
        }
        _txFormatEnum = parseSampleFormat(format);
        _txChannels   = channels.empty() ? std::vector<size_t>{0} : channels;
        return reinterpret_cast<SoapySDR::Stream*>(&_txStreamSentinel);
    }

    void closeStream(SoapySDR::Stream* /*stream*/) override {}

    int activateStream(SoapySDR::Stream* stream, const int /*flags*/ = 0, const long long /*timeNs*/ = 0, const size_t /*numElems*/ = 0) override {
        if (stream == reinterpret_cast<SoapySDR::Stream*>(&_rxStreamSentinel)) {
            _rxStreamActive.store(true, std::memory_order_relaxed);
            _lastReadTime = std::chrono::steady_clock::now();
        } else {
            _txStreamActive.store(true, std::memory_order_relaxed);
        }
        return 0;
    }

    int deactivateStream(SoapySDR::Stream* stream, const int /*flags*/ = 0, const long long /*timeNs*/ = 0) override {
        if (stream == reinterpret_cast<SoapySDR::Stream*>(&_rxStreamSentinel)) {
            _rxStreamActive.store(false, std::memory_order_relaxed);
            drainBuffers();
        } else {
            _txStreamActive.store(false, std::memory_order_relaxed);
        }
        return 0;
    }

    int writeStream(SoapySDR::Stream* /*stream*/, const void* const* buffs, const size_t numElems, int& /*flags*/, const long long /*timeNs*/ = 0, const long /*timeoutUs*/ = 100000) override {
        if (!_txStreamActive.load(std::memory_order_relaxed)) {
            return SOAPY_SDR_STREAM_ERROR;
        }
        if (_deviceMode == DeviceMode::TxOnly) {
            return static_cast<int>(numElems); // null sink — accept and discard
        }

        std::size_t nWritten = numElems;
        for (std::size_t chIdx = 0UZ; chIdx < _txChannels.size(); ++chIdx) {
            auto ch = _txChannels[chIdx];
            if (ch >= _numChannels) {
                continue;
            }

            auto& state = *_channels[ch];

            auto writerSpan = state.rxWriter.tryReserve<gr::SpanReleasePolicy::ProcessNone>(numElems);
            if (writerSpan.empty()) {
                return SOAPY_SDR_TIMEOUT; // backpressure — RX buffer full
            }
            auto nWrite = std::min(writerSpan.size(), numElems);
            nWritten    = std::min(nWritten, nWrite);

            std::span<const CF32> txSpan;
            if (_txFormatEnum == SampleFormat::cf32) {
                txSpan = {static_cast<const CF32*>(buffs[chIdx]), nWrite};
            } else {
                state.txScratch.resize(nWrite);
                convertToComplex(buffs[chIdx], _txFormatEnum, std::span<CF32>(state.txScratch).first(nWrite));
                txSpan = std::span<const CF32>(state.txScratch).first(nWrite);
            }

            std::span<CF32> outSpan(writerSpan.begin(), nWrite);
            state.model.process(txSpan, outSpan);
            writerSpan.publish(nWrite);
        }
        return static_cast<int>(nWritten);
    }

    int readStream(SoapySDR::Stream* /*stream*/, void* const* buffs, const size_t numElems, int& flags, long long& timeNs, const long timeoutUs = 100000) override {
        if (!_rxStreamActive.load(std::memory_order_relaxed)) {
            return SOAPY_SDR_STREAM_ERROR;
        }

        flags  = 0;
        timeNs = 0;

        if (_simulateTiming.load(std::memory_order_relaxed) && !_rxChannels.empty()) {
            auto ch       = _rxChannels[0];
            auto rate     = (ch < _numChannels) ? _channels[ch]->sampleRate : 1e6;
            auto now      = std::chrono::steady_clock::now();
            auto elapsed  = std::chrono::duration<double>(now - _lastReadTime).count();
            auto expected = static_cast<double>(numElems) / rate;
            if (elapsed < expected) {
                if (std::chrono::duration<double>(expected - elapsed) > std::chrono::microseconds(timeoutUs)) {
                    return SOAPY_SDR_TIMEOUT;
                }
                std::this_thread::sleep_for(std::chrono::duration<double>(expected - elapsed));
            }
            _lastReadTime = std::chrono::steady_clock::now();
        }

        if (_deviceMode == DeviceMode::RxOnly) {
            return readStreamRxOnly(buffs, numElems);
        }

        std::size_t available = numElems;
        for (auto ch : _rxChannels) {
            if (ch < _numChannels) {
                auto avail = _channels[ch]->rxReader.available();
                if (avail <= 0) {
                    return SOAPY_SDR_TIMEOUT;
                }
                available = std::min(available, static_cast<std::size_t>(avail));
            }
        }
        if (available == 0UZ) {
            return SOAPY_SDR_TIMEOUT;
        }

        auto nRead = std::min(available, numElems);
        for (std::size_t chIdx = 0UZ; chIdx < _rxChannels.size(); ++chIdx) {
            auto ch = _rxChannels[chIdx];
            if (ch >= _numChannels) {
                continue;
            }

            auto& state   = _channels[ch];
            auto  rxSpan  = state->rxReader.get(nRead);
            auto  nActual = std::min(rxSpan.size(), nRead);
            convertFromComplex(std::span<const CF32>(rxSpan.begin(), nActual), buffs[chIdx], _rxFormatEnum);
            if (!rxSpan.consume(nActual)) {
                return SOAPY_SDR_STREAM_ERROR;
            }
        }
        return static_cast<int>(nRead);
    }

    int readStreamStatus(SoapySDR::Stream* /*stream*/, size_t& /*chanMask*/, int& /*flags*/, long long& /*timeNs*/, const long /*timeoutUs*/ = 100000) override { return SOAPY_SDR_TIMEOUT; }

    std::vector<std::string> listTimeSources() const override { return {"none"}; }
    std::string              getTimeSource() const override { return "none"; }
    void                     setTimeSource(const std::string& /*source*/) override {}
    bool                     hasHardwareTime(const std::string& /*what*/ = "") const override { return false; }
    long long                getHardwareTime(const std::string& /*what*/ = "") const override { return 0; }
    void                     setHardwareTime(const long long /*timeNs*/, const std::string& /*what*/ = "") override {}
    void                     setCommandTime(const long long /*timeNs*/, const std::string& /*what*/ = "") override {}

    void                     setMasterClockRate(const double rate) override { _masterClockRate = rate; }
    double                   getMasterClockRate() const override { return _masterClockRate; }
    SoapySDR::RangeList      getMasterClockRates() const override { return {SoapySDR::Range(0.0, 100e6)}; }
    void                     setReferenceClockRate(const double rate) override { _referenceClockRate = rate; }
    double                   getReferenceClockRate() const override { return _referenceClockRate; }
    SoapySDR::RangeList      getReferenceClockRates() const override { return {SoapySDR::Range(0.0, 100e6)}; }
    std::vector<std::string> listClockSources() const override { return {"internal", "external"}; }
    std::string              getClockSource() const override { return _clockSource; }
    void                     setClockSource(const std::string& source) override { _clockSource = source; }

    std::vector<std::string> listSensors() const override { return {"temperature", "lo_locked"}; }
    SoapySDR::ArgInfo        getSensorInfo(const std::string& key) const override {
        SoapySDR::ArgInfo info;
        info.key = key;
        if (key == "temperature") {
            info.value       = "25.0";
            info.type        = SoapySDR::ArgInfo::FLOAT;
            info.description = "simulated board temperature in degrees C";
        } else if (key == "lo_locked") {
            info.value       = "true";
            info.type        = SoapySDR::ArgInfo::BOOL;
            info.description = "simulated LO lock indicator";
        }
        return info;
    }
    std::string readSensor(const std::string& key) const override {
        if (key == "temperature") {
            return "25.0";
        }
        if (key == "lo_locked") {
            return "true";
        }
        return "";
    }
    std::vector<std::string> listSensors(const int /*direction*/, const size_t /*channel*/) const override { return {"rssi"}; }
    SoapySDR::ArgInfo        getSensorInfo(const int /*direction*/, const size_t /*channel*/, const std::string& key) const override {
        SoapySDR::ArgInfo info;
        info.key = key;
        if (key == "rssi") {
            info.value       = "-60.0";
            info.type        = SoapySDR::ArgInfo::FLOAT;
            info.description = "simulated received signal strength in dBm";
        }
        return info;
    }
    std::string readSensor(const int /*direction*/, const size_t /*channel*/, const std::string& key) const override {
        if (key == "rssi") {
            return "-60.0";
        }
        return "";
    }

    std::vector<std::string> listRegisterInterfaces() const override { return {"loopback_regs"}; }
    void                     writeRegister(const std::string& name, const unsigned addr, const unsigned value) override { _registers[{name, addr}] = value; }
    unsigned                 readRegister(const std::string& name, const unsigned addr) const override {
        if (auto it = _registers.find({name, addr}); it != _registers.end()) {
            return it->second;
        }
        return 0;
    }
    void     writeRegister(const unsigned addr, const unsigned value) override { writeRegister("", addr, value); }
    unsigned readRegister(const unsigned addr) const override { return readRegister("", addr); }
    void     writeRegisters(const std::string& name, const unsigned addr, const std::vector<unsigned>& value) override {
        for (std::size_t i = 0UZ; i < value.size(); ++i) {
            writeRegister(name, addr + static_cast<unsigned>(i), value[i]);
        }
    }
    std::vector<unsigned> readRegisters(const std::string& name, const unsigned addr, const size_t length) const override {
        std::vector<unsigned> result(length);
        for (std::size_t i = 0UZ; i < length; ++i) {
            result[i] = readRegister(name, addr + static_cast<unsigned>(i));
        }
        return result;
    }

    std::vector<std::string> listGPIOBanks() const override { return {"MAIN"}; }
    void                     writeGPIO(const std::string& bank, const unsigned value) override { _gpioValues[bank] = value; }
    void                     writeGPIO(const std::string& bank, const unsigned value, const unsigned mask) override { _gpioValues[bank] = (_gpioValues[bank] & ~mask) | (value & mask); }
    unsigned                 readGPIO(const std::string& bank) const override {
        if (auto it = _gpioValues.find(bank); it != _gpioValues.end()) {
            return it->second;
        }
        return 0;
    }
    void     writeGPIODir(const std::string& bank, const unsigned dir) override { _gpioDirValues[bank] = dir; }
    void     writeGPIODir(const std::string& bank, const unsigned dir, const unsigned mask) override { _gpioDirValues[bank] = (_gpioDirValues[bank] & ~mask) | (dir & mask); }
    unsigned readGPIODir(const std::string& bank) const override {
        if (auto it = _gpioDirValues.find(bank); it != _gpioDirValues.end()) {
            return it->second;
        }
        return 0;
    }

    void        writeI2C(const int /*addr*/, const std::string& /*data*/) override {}
    std::string readI2C(const int /*addr*/, const size_t numBytes) override { return std::string(numBytes, '\0'); }
    unsigned    transactSPI(const int /*addr*/, const unsigned data, const size_t /*numBits*/) override { return data; }

    std::vector<std::string> listUARTs() const override { return {}; }
    void                     writeUART(const std::string& /*which*/, const std::string& /*data*/) override {}
    std::string              readUART(const std::string& /*which*/, const long /*timeoutUs*/ = 100000) const override { return ""; }

    void* getNativeDeviceHandle() const override { return nullptr; }

    void writeSetting(const std::string& key, const std::string& value) override {
        if (key == "simulate_timing") {
            _simulateTiming.store(value == "true" || value == "1", std::memory_order_relaxed);
            return;
        }
        if (key == "device_mode") {
            _deviceMode = parseDeviceMode(value);
            if (_deviceMode == DeviceMode::RxOnly) {
                _simulateTiming.store(true, std::memory_order_relaxed);
            }
            return;
        }
        auto model = modelFromSetting(key, value);
        if (model.process) {
            assert(!_txStreamActive.load(std::memory_order_relaxed) && "writeSetting(model) must be called before activateStream(TX)");
            setChannelModel(std::move(model));
        }
    }

    std::string readSetting(const std::string& key) const override {
        if (key == "simulate_timing") {
            return _simulateTiming.load(std::memory_order_relaxed) ? "true" : "false";
        }
        if (key == "device_mode") {
            switch (_deviceMode) {
            case DeviceMode::RxOnly: return "rx_only";
            case DeviceMode::TxOnly: return "tx_only";
            default: return "loopback";
            }
        }
        return "";
    }

    void writeSetting(const int /*direction*/, const size_t channel, const std::string& key, const std::string& value) override {
        if (channel >= _numChannels) {
            return;
        }
        auto model = modelFromSetting(key, value);
        if (model.process) {
            assert(!_txStreamActive.load(std::memory_order_relaxed) && "writeSetting(model) must be called before activateStream(TX)");
            _channels[channel]->model = std::move(model);
        }
    }

    std::string readSetting(const int /*direction*/, const size_t /*channel*/, const std::string& /*key*/) const override { return ""; }

    SoapySDR::ArgInfo getSettingInfo(const std::string& key) const override {
        for (const auto& info : getSettingInfo()) {
            if (info.key == key) {
                return info;
            }
        }
        return {};
    }

    SoapySDR::ArgInfo getSettingInfo(const int direction, const size_t channel, const std::string& key) const override {
        for (const auto& info : getSettingInfo(direction, channel)) {
            if (info.key == key) {
                return info;
            }
        }
        return {};
    }

    SoapySDR::ArgInfoList getSettingInfo(const int /*direction*/, const size_t /*channel*/) const override {
        SoapySDR::ArgInfoList infos;
        {
            SoapySDR::ArgInfo info;
            info.key         = "channel_model";
            info.value       = "passthrough";
            info.type        = SoapySDR::ArgInfo::STRING;
            info.description = "per-channel model type";
            info.options     = {"passthrough"};
            infos.push_back(info);
        }
        {
            SoapySDR::ArgInfo info;
            info.key         = "attenuation_dB";
            info.value       = "0";
            info.type        = SoapySDR::ArgInfo::FLOAT;
            info.description = "per-channel attenuation in dB (negative values attenuate)";
            infos.push_back(info);
        }
        return infos;
    }

    SoapySDR::ArgInfoList getSettingInfo() const override {
        SoapySDR::ArgInfoList infos;
        {
            SoapySDR::ArgInfo info;
            info.key         = "device_mode";
            info.value       = "loopback";
            info.type        = SoapySDR::ArgInfo::STRING;
            info.description = "operating mode: loopback (TX→RX), rx_only (tone→RX), tx_only (TX→null)";
            info.options     = {"loopback", "rx_only", "tx_only"};
            infos.push_back(info);
        }
        {
            SoapySDR::ArgInfo info;
            info.key         = "simulate_timing";
            info.value       = "false";
            info.type        = SoapySDR::ArgInfo::BOOL;
            info.description = "rate-limit readStream to simulate real sample rate";
            infos.push_back(info);
        }
        {
            SoapySDR::ArgInfo info;
            info.key         = "channel_model";
            info.value       = "passthrough";
            info.type        = SoapySDR::ArgInfo::STRING;
            info.description = "channel model type (applies to all channels)";
            info.options     = {"passthrough"};
            infos.push_back(info);
        }
        return infos;
    }

private:
    static DeviceMode parseDeviceMode(const std::string& value) {
        if (value == "rx_only") {
            return DeviceMode::RxOnly;
        }
        if (value == "tx_only") {
            return DeviceMode::TxOnly;
        }
        return DeviceMode::Loopback;
    }

    static void generateTone(ChannelState& state, std::span<CF32> out) {
        double phaseInc = 2.0 * std::numbers::pi * state.frequency / state.sampleRate;
        for (auto& sample : out) {
            sample = CF32{static_cast<float>(std::cos(state.rxPhase)), static_cast<float>(std::sin(state.rxPhase))};
            state.rxPhase += phaseInc;
        }
        state.rxPhase = std::fmod(state.rxPhase, 2.0 * std::numbers::pi);
    }

    int readStreamRxOnly(void* const* buffs, std::size_t numElems) {
        for (std::size_t chIdx = 0UZ; chIdx < _rxChannels.size(); ++chIdx) {
            auto ch = _rxChannels[chIdx];
            if (ch >= _numChannels) {
                continue;
            }
            auto& state = *_channels[ch];

            _rxToneScratch.resize(numElems);
            generateTone(state, _rxToneScratch);

            _rxModelScratch.resize(numElems);
            state.model.process(_rxToneScratch, _rxModelScratch);

            convertFromComplex(std::span<const CF32>(_rxModelScratch).first(numElems), buffs[chIdx], _rxFormatEnum);
        }
        return static_cast<int>(numElems);
    }

    static SampleFormat parseSampleFormat(const std::string& format) {
        if (format == SOAPY_SDR_CS16) {
            return SampleFormat::cs16;
        }
        if (format == SOAPY_SDR_CU8) {
            return SampleFormat::cu8;
        }
        return SampleFormat::cf32;
    }

    static ChannelModel modelFromSetting(const std::string& key, const std::string& value) {
        if (key == "channel_model") {
            return ChannelModel::passthrough();
        }
        if (key == "attenuation_dB") {
            float dB = 0.f;
            std::from_chars(value.data(), value.data() + value.size(), dB);
            return ChannelModel::attenuation(dB);
        }
        if (key == "noise_floor_dBFS") {
            float dBFS = 0.f;
            std::from_chars(value.data(), value.data() + value.size(), dBFS);
            return ChannelModel::awgn(dBFS);
        }
        if (key == "delay_samples") {
            std::size_t n = 0UZ;
            std::from_chars(value.data(), value.data() + value.size(), n);
            return ChannelModel::delay(n);
        }
        return {};
    }

    void drainBuffers() {
        for (auto& ch : _channels) {
            auto avail = ch->rxReader.available();
            if (avail > 0) {
                auto                  span = ch->rxReader.get(static_cast<std::size_t>(avail));
                [[maybe_unused]] auto ok   = span.consume(static_cast<std::size_t>(avail));
            }
        }
    }

    static void convertToComplex(const void* src, SampleFormat format, std::span<CF32> out) {
        switch (format) {
        case SampleFormat::cf32: std::memcpy(out.data(), src, out.size() * sizeof(CF32)); break;
        case SampleFormat::cs16: {
            const auto* s = static_cast<const int16_t*>(src);
            for (std::size_t i = 0UZ; i < out.size(); ++i) {
                out[i] = {static_cast<float>(s[2 * i]) / 32768.f, static_cast<float>(s[2 * i + 1]) / 32768.f};
            }
        } break;
        case SampleFormat::cu8: {
            const auto* s = static_cast<const uint8_t*>(src);
            for (std::size_t i = 0UZ; i < out.size(); ++i) {
                out[i] = {(static_cast<float>(s[2 * i]) - 127.5f) / 127.5f, (static_cast<float>(s[2 * i + 1]) - 127.5f) / 127.5f};
            }
        } break;
        }
    }

    static void convertFromComplex(std::span<const CF32> in, void* dst, SampleFormat format) {
        switch (format) {
        case SampleFormat::cf32: std::memcpy(dst, in.data(), in.size() * sizeof(CF32)); break;
        case SampleFormat::cs16: {
            auto* d = static_cast<int16_t*>(dst);
            for (std::size_t i = 0UZ; i < in.size(); ++i) {
                d[2 * i]     = static_cast<int16_t>(std::clamp(in[i].real() * 32767.f, -32768.f, 32767.f));
                d[2 * i + 1] = static_cast<int16_t>(std::clamp(in[i].imag() * 32767.f, -32768.f, 32767.f));
            }
        } break;
        case SampleFormat::cu8: {
            auto* d = static_cast<uint8_t*>(dst);
            for (std::size_t i = 0UZ; i < in.size(); ++i) {
                d[2 * i]     = static_cast<uint8_t>(std::clamp(in[i].real() * 127.5f + 127.5f, 0.f, 255.f));
                d[2 * i + 1] = static_cast<uint8_t>(std::clamp(in[i].imag() * 127.5f + 127.5f, 0.f, 255.f));
            }
        } break;
        }
    }
};

inline std::shared_ptr<LoopbackDevice> DeviceRegistry::findOrCreate(std::size_t instanceId, const SoapySDR::Kwargs& args) {
    auto  lock = std::lock_guard(mutex());
    auto& map  = instances();
    if (auto it = map.find(instanceId); it != map.end()) {
        if (auto existing = it->second.lock()) {
            return existing;
        }
        map.erase(it);
    }
    auto device     = std::make_shared<LoopbackDevice>(args);
    map[instanceId] = device;
    return device;
}

} // namespace gr::blocks::sdr::loopback

#endif // GNURADIO_LOOPBACK_DEVICE_HPP
