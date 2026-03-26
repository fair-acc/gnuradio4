#ifndef SOAPYRAIIWRAPPER_HPP
#define SOAPYRAIIWRAPPER_HPP

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-W#warnings"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#endif
#include <SoapySDR/Device.h> // using SoapySDR's C-API as intermediate interface to mitigate ABI-issues between stdlibc++ and libc++
#include <SoapySDR/Formats.h>
#include <SoapySDR/Modules.h>
#include <SoapySDR/Version.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <cassert>
#include <cstring>
#include <expected>
#include <format>
#include <map>
#include <memory>
#include <source_location>
#include <sstream>
#include <string>
#include <vector>

#include <gnuradio-4.0/meta/formatter.hpp>

namespace gr::blocks::sdr::soapy {

using Range      = SoapySDRRange;
using Kwargs     = std::map<std::string, std::string>;
using KwargsList = std::vector<Kwargs>;

struct ArgInfo { // redeclare to be trivially constructable and ABI compatible
    std::string key;
    std::string value;
    std::string name;
    std::string description;
    std::string units;
    enum Type { BOOL, INT, FLOAT, STRING } type = STRING;
    Range                    range{};
    std::vector<std::string> options;
    std::vector<std::string> optionNames;
};

namespace detail {

class KwargsWrapper {
    SoapySDRKwargs _cArgs;
    bool           _valid = true;

public:
    KwargsWrapper(const Kwargs& args) {
        std::memset(&_cArgs, 0, sizeof(_cArgs));
        for (const auto& it : args) {
            if (SoapySDRKwargs_set(&_cArgs, it.first.c_str(), it.second.c_str()) != 0) {
                _valid = false;
                break;
            }
        }
    }

    ~KwargsWrapper() { SoapySDRKwargs_clear(&_cArgs); }

    [[nodiscard]] bool            valid() const noexcept { return _valid; }
    [[nodiscard]] SoapySDRKwargs* get() noexcept { return &_cArgs; }
    operator const SoapySDRKwargs*() const noexcept { return &_cArgs; }
    operator SoapySDRKwargs*() noexcept { return &_cArgs; }
};

[[nodiscard]] inline KwargsList convertToCpp(SoapySDRKwargs* results, std::size_t length) {
    KwargsList kwargsList;
    kwargsList.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
        Kwargs kwargs;
        for (std::size_t j = 0; j < results[i].size; ++j) {
            kwargs[results[i].keys[j]] = results[i].vals[j];
        }
        kwargsList.push_back(kwargs);
    }
    SoapySDRKwargsList_clear(results, length);
    return kwargsList;
}

[[nodiscard]] inline std::vector<std::string> convertToCpp(char** strings, std::size_t length) {
    std::vector<std::string> stringList;
    stringList.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
        stringList.emplace_back(std::string(strings[i]));
    }
    SoapySDRStrings_clear(&strings, length);
    return stringList;
}

[[nodiscard]] inline std::vector<ArgInfo> convertToCpp(SoapySDRArgInfo* infos, std::size_t size) {
    std::vector<ArgInfo> argInfoList;
    argInfoList.resize(size); // Preallocate space

    for (std::size_t i = 0; i < size; ++i) {
        SoapySDRArgInfo& cInfo   = infos[i];
        ArgInfo&         argInfo = argInfoList[i];

        argInfo.key         = cInfo.key ? std::string(cInfo.key) : "";
        argInfo.value       = cInfo.value ? std::string(cInfo.value) : "";
        argInfo.name        = cInfo.name ? std::string(cInfo.name) : "";
        argInfo.description = cInfo.description ? std::string(cInfo.description) : "";
        argInfo.units       = cInfo.units ? std::string(cInfo.units) : "";
        argInfo.type        = static_cast<ArgInfo::Type>(cInfo.type);
        argInfo.range       = {cInfo.range.minimum, cInfo.range.maximum, cInfo.range.step};

        if (cInfo.options != nullptr) {
            argInfo.options.resize(cInfo.numOptions);
            for (std::size_t j = 0; j < cInfo.numOptions; ++j) {
                argInfo.options[j] = cInfo.options[j] ? std::string(cInfo.options[j]) : "";
            }
        }

        if (cInfo.optionNames != nullptr) {
            argInfo.optionNames.resize(cInfo.numOptions);
            for (std::size_t j = 0; j < cInfo.numOptions; ++j) {
                argInfo.optionNames[j] = cInfo.optionNames[j] ? std::string(cInfo.optionNames[j]) : "";
            }
        }
    }
    SoapySDRArgInfoList_clear(infos, size);

    return argInfoList;
}

template<typename T>
constexpr const char* toSoapySDRFormat() {
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return SOAPY_SDR_CF64;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return SOAPY_SDR_CF32;
    } else if constexpr (std::is_same_v<T, double>) {
        return SOAPY_SDR_F64;
    } else if constexpr (std::is_same_v<T, float>) {
        return SOAPY_SDR_F32;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return SOAPY_SDR_S32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return SOAPY_SDR_U32;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return SOAPY_SDR_S16;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return SOAPY_SDR_U16;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return SOAPY_SDR_S8;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return SOAPY_SDR_U8;
    } else {
        static_assert(!gr::meta::always_false<T>, "snsupported type");
    }
}

[[nodiscard]] inline std::string getSoapyFlagNames(int flags) {
    using namespace std::string_literals;
    std::vector<std::string> flagNames;

    if (flags & SOAPY_SDR_END_BURST) {
        flagNames.emplace_back("END_BURST"s);
    }
    if (flags & SOAPY_SDR_HAS_TIME) {
        flagNames.emplace_back("HAS_TIME"s);
    }
    if (flags & SOAPY_SDR_END_ABRUPT) {
        flagNames.emplace_back("END_ABRUPT"s);
    }
    if (flags & SOAPY_SDR_ONE_PACKET) {
        flagNames.emplace_back("ONE_PACKET"s);
    }
    if (flags & SOAPY_SDR_MORE_FRAGMENTS) {
        flagNames.emplace_back("MORE_FRAGMENTS"s);
    }
    if (flags & SOAPY_SDR_WAIT_TRIGGER) {
        flagNames.emplace_back("WAIT_TRIGGER"s);
    }

    return std::format("{}", gr::join(flagNames, ", "));
}

inline void printSoapyReturnDebugInfo(int ret, int flags, long long time_ns) {
    if (ret == SOAPY_SDR_TIMEOUT) {
        std::print("TIMEOUT - ");
    } else if (ret < 0) {
        std::print("ERROR ({}) - '{}' - ", ret, SoapySDR_errToStr(ret));
    }
    std::println("ret = {}, flags({}) = [{}], time_ns = {}", ret, flags, getSoapyFlagNames(flags), time_ns);
}

} // namespace detail

[[nodiscard]] inline std::vector<std::string> getSoapySDRModules() {
    std::size_t moduleCount = 0UZ;
    char**      modules     = SoapySDR_listModules(&moduleCount);
    return detail::convertToCpp(modules, moduleCount);
}

/**
 * @brief C++ SoapySDR::Device-like via C-API wrapper
 *
 * This is a workaround to ensure ABI compatibility for GR4 being compiled
 * with libc++ and the Soapy wrapper with GCC and vice-versa.
 *
 * Implements only a subset of SoapySDR as much as it's needed for the GR4 Soapy block
 */
class Device {
    std::shared_ptr<SoapySDRDevice> _device{nullptr};

public:
    static KwargsList enumerate(const Kwargs& args = Kwargs()) {
        std::size_t     length  = 0UZ;
        SoapySDRKwargs* results = SoapySDRDevice_enumerate(detail::KwargsWrapper(args), &length);
        return detail::convertToCpp(results, length);
    }

    static KwargsList enumerate(const std::string& args) {
        Kwargs            kwargs;
        std::stringstream ss(args);
        std::string       keyVal;
        while (std::getline(ss, keyVal, ',')) {
            auto pos = keyVal.find('=');
            if (pos != std::string::npos) {
                auto key    = keyVal.substr(0, pos);
                auto val    = keyVal.substr(pos + 1);
                kwargs[key] = val;
            }
        }
        return enumerate(kwargs);
    }

    Device() = default;

    Device(const Kwargs& args, std::source_location location = std::source_location::current()) {
        if (std::string_view(SOAPY_SDR_ABI_VERSION) != std::string_view(SoapySDR_getABIVersion())) {
            throw gr::exception(std::format("SoapySDR ABI mismatch: this {} vs. library {}", SOAPY_SDR_ABI_VERSION, SoapySDR_getABIVersion()), location);
        }

        _device.reset(SoapySDRDevice_make(detail::KwargsWrapper(args)), [](SoapySDRDevice* device) {
            if (device == nullptr) {
                return;
            }
            if (int ret = SoapySDRDevice_unmake(device); ret) {
                std::println(stderr, "[SoapySDR] error({}) closing device: '{}'", ret, SoapySDR_errToStr(ret));
            }
        });
        if (!_device) {
            throw gr::exception(std::format("Device({}) - SoapySDRDevice_make failed", args), location);
        }
    }

    static std::expected<Device, std::string> make(const Kwargs& args = Kwargs()) {
        if (std::string_view(SOAPY_SDR_ABI_VERSION) != std::string_view(SoapySDR_getABIVersion())) {
            return std::unexpected(std::format("SoapySDR ABI mismatch: this {} vs. library {}", SOAPY_SDR_ABI_VERSION, SoapySDR_getABIVersion()));
        }
        detail::KwargsWrapper cArgs(args);
        if (!cArgs.valid()) {
            return std::unexpected(std::format("SoapySDRKwargs_set allocation failed for {}", args));
        }
        SoapySDRDevice* raw = SoapySDRDevice_make(cArgs);
        if (!raw) {
            return std::unexpected(std::format("SoapySDRDevice_make failed for {}", args));
        }
        Device dev;
        dev._device.reset(raw, [](SoapySDRDevice* device) {
            if (device) {
                SoapySDRDevice_unmake(device);
            }
        });
        return dev;
    }

    Device(const Device& other) : _device(other._device) {}
    Device(Device&& other) noexcept : _device(std::move(other._device)) { other._device = nullptr; }
    Device& operator=(Device&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        _device       = std::move(other._device);
        other._device = nullptr;
        return *this;
    }

    void            reset() { _device.reset(); }
    SoapySDRDevice* get() const { return _device.get(); }

    std::vector<std::string> listAvailableAntennas(int direction, std::size_t channel) const {
        std::size_t length   = 0UZ;
        char**      antennas = SoapySDRDevice_listAntennas(_device.get(), direction, channel, &length);
        return detail::convertToCpp(antennas, length);
    }

    std::expected<void, std::string> setAntenna(int direction, std::size_t channel, std::string_view antennaName) {
        if (int error = SoapySDRDevice_setAntenna(_device.get(), direction, channel, antennaName.data()); error) {
            return std::unexpected(std::format("setAntenna({}, {}, {}) error({}): {}", direction, channel, antennaName, error, SoapySDR_errToStr(error)));
        }
        return {};
    }

    std::string getAntenna(int direction, std::size_t channel) const {
        char*       value = SoapySDRDevice_getAntenna(_device.get(), direction, channel);
        std::string result(value);
        free(value);
        return result;
    }

    std::vector<std::string> listAvailableGainElements(int direction, std::size_t channel) const {
        std::size_t length = 0UZ;
        char**      gains  = SoapySDRDevice_listGains(_device.get(), direction, channel, &length);
        return detail::convertToCpp(gains, length);
    }

    bool hasAutomaticGainControl(int direction, std::size_t channel) const { return SoapySDRDevice_hasGainMode(_device.get(), direction, channel); }

    std::expected<void, std::string> setAutomaticGainControl(int direction, std::size_t channel, bool automatic) {
        if (int error = SoapySDRDevice_setGainMode(_device.get(), direction, channel, automatic); error) {
            return std::unexpected(std::format("setAutomaticGainControl({}, {}, {}) error({}): {}", direction, channel, automatic, error, SoapySDR_errToStr(error)));
        }
        return {};
    }

    bool isAutomaticGainControl(int direction, std::size_t channel) { return SoapySDRDevice_getGainMode(_device.get(), direction, channel); }

    std::expected<void, std::string> setGain(int direction, std::size_t channel, double gain_dB, std::string_view gainElement = {}) {
        int error = gainElement.empty() ? SoapySDRDevice_setGain(_device.get(), direction, channel, gain_dB) : SoapySDRDevice_setGainElement(_device.get(), direction, channel, gainElement.data(), gain_dB);
        if (error) {
            return std::unexpected(std::format("setGain({}, {}, {}) error({}): {}", direction, channel, gain_dB, error, SoapySDR_errToStr(error)));
        }
        return {};
    }

    double getGain(int direction, std::size_t channel, std::string_view gainElement = {}) const {
        if (gainElement.empty()) {
            return SoapySDRDevice_getGain(_device.get(), direction, channel);
        }
        return SoapySDRDevice_getGainElement(_device.get(), direction, channel, gainElement.data());
    }

    std::vector<double> listAvailableBandwidths(int direction, std::size_t channel) const {
        std::size_t         length     = 0UZ;
        double*             bandwidths = SoapySDRDevice_listBandwidths(_device.get(), direction, channel, &length);
        std::vector<double> bwList(bandwidths, bandwidths + length);
        free(bandwidths);
        return bwList;
    }

    std::expected<void, std::string> setBandwidth(int direction, std::size_t channel, double bandwidthHz) {
        if (int error = SoapySDRDevice_setBandwidth(_device.get(), direction, channel, bandwidthHz); error) {
            return std::unexpected(std::format("setBandwidth({}, {}, {}) error({}): {}", direction, channel, bandwidthHz, error, SoapySDR_errToStr(error)));
        }
        return {};
    }

    double getBandwidth(int direction, std::size_t channel) const { return SoapySDRDevice_getBandwidth(_device.get(), direction, channel); }

    std::vector<Range> getOverallFrequencyRange(int direction, std::size_t channel) const {
        std::size_t    count;
        SoapySDRRange* ranges = SoapySDRDevice_getFrequencyRange(_device.get(), direction, channel, &count);

        std::vector<Range> rangeList;
        rangeList.reserve(count);

        for (std::size_t i = 0; i < count; ++i) {
            rangeList.emplace_back(ranges[i].minimum, ranges[i].maximum, ranges[i].step);
        }
        free(ranges);
        return rangeList;
    }

    std::expected<void, std::string> setCenterFrequency(int direction, std::size_t channel, double frequency, const Kwargs& args = Kwargs()) {
        if (int error = SoapySDRDevice_setFrequency(_device.get(), direction, channel, frequency, detail::KwargsWrapper(args)); error) {
            return std::unexpected(std::format("setCenterFrequency({}, {}, {}) error({}): {}", direction, channel, frequency, error, SoapySDR_errToStr(error)));
        }
        return {};
    }

    double getCenterFrequency(int direction, std::size_t channel) const { return SoapySDRDevice_getFrequency(_device.get(), direction, channel); }

    std::vector<double> listSampleRates(int direction, std::size_t channel) const {
        std::size_t         length = 0UZ;
        double*             rates  = SoapySDRDevice_listSampleRates(_device.get(), direction, channel, &length);
        std::vector<double> rateList(rates, rates + length);
        free(rates);
        return rateList;
    }

    std::expected<void, std::string> setSampleRate(int direction, std::size_t channel, double rate) {
        if (int error = SoapySDRDevice_setSampleRate(_device.get(), direction, channel, rate); error) {
            return std::unexpected(std::format("setSampleRate({}, {}, {}) error({}): {}", direction, channel, rate, error, SoapySDR_errToStr(error)));
        }
        return {};
    }

    double getSampleRate(int direction, std::size_t channel) const { return SoapySDRDevice_getSampleRate(_device.get(), direction, channel); }

    std::vector<std::string> listAvailableTimeSources() const {
        std::size_t length  = 0UZ;
        char**      sources = SoapySDRDevice_listTimeSources(_device.get(), &length);
        return detail::convertToCpp(sources, length);
    }

    std::string getTimeSource() const {
        char*       value = SoapySDRDevice_getTimeSource(_device.get());
        std::string result(value);
        free(value);
        return result;
    }

    std::expected<void, std::string> setTimeSource(const std::string& source) {
        if (int error = SoapySDRDevice_setTimeSource(_device.get(), source.c_str()); error) {
            return std::unexpected(std::format("setTimeSource({}) error({}): {}", source, error, SoapySDR_errToStr(error)));
        }
        return {};
    }

    std::uint64_t getHardwareTime(std::string_view what = {}) const { return static_cast<std::uint64_t>(SoapySDRDevice_getHardwareTime(_device.get(), what.data())); }

    std::expected<void, std::string> setHardwareTime(long long timeNs, const std::string& event = "") {
        if (int error = SoapySDRDevice_setHardwareTime(_device.get(), timeNs, event.empty() ? nullptr : event.c_str()); error) {
            return std::unexpected(std::format("setHardwareTime({}) error({}): {}", timeNs, error, SoapySDR_errToStr(error)));
        }
        return {};
    }

    double getMasterClockRate() const { return SoapySDRDevice_getMasterClockRate(_device.get()); }

    template<typename TValueType, int direction>
    class Stream {
        std::shared_ptr<SoapySDRDevice> _device{nullptr};
        std::shared_ptr<SoapySDRStream> _stream{nullptr};

    protected:
        Stream(std::shared_ptr<SoapySDRDevice> device, SoapySDRStream* cStream)
            : _device(device), _stream(cStream, [device](SoapySDRStream* stream) {
                  if (device.get() == nullptr || stream == nullptr) {
                      return;
                  }
                  if (int ret = SoapySDRDevice_closeStream(device.get(), stream); ret) {
                      std::println(stderr, "[SoapySDR] error({}) closing stream: '{}'", ret, SoapySDR_errToStr(ret));
                  }
              }) {}
        friend class Device;

    public:
        using value_type                       = TValueType;
        constexpr static std::size_t Direction = direction;

        Stream() noexcept                = default;
        Stream(const Stream&)            = delete;
        Stream& operator=(const Stream&) = delete;
        Stream(Stream&& other) noexcept : Stream() {
            swap(*this, other);
            other._stream = nullptr;
            other._device = nullptr;
        }

        Stream& operator=(Stream&& other) noexcept {
            swap(*this, other);
            return *this;
        }

        friend void swap(Stream& first, Stream& second) noexcept {
            using std::swap;
            swap(first._stream, second._stream);
            swap(first._device, second._device);
        }

        void reset() {
            _device.reset();
            _stream.reset();
        }

        std::expected<void, std::string> activate(int flags = 0, long long timeNs = 0, std::size_t numElems = 0UZ) {
            int ret = SoapySDRDevice_activateStream(_device.get(), _stream.get(), flags, timeNs, numElems);
            if (ret != 0) {
                return std::unexpected(std::format("activate(flags={}, timeNs={}, numElems={}) error({}): {}", flags, timeNs, numElems, ret, SoapySDR_errToStr(ret)));
            }
            return {};
        }

        std::expected<void, std::string> deactivate(int flags = 0, long long timeNs = 0) {
            int ret = SoapySDRDevice_deactivateStream(_device.get(), _stream.get(), flags, timeNs);
            if (ret != 0) {
                return std::unexpected(std::format("deactivate(flags={}, timeNs={}) error({}): {}", flags, timeNs, ret, SoapySDR_errToStr(ret)));
            }
            return {};
        }

        template<typename... TBuffers>
        requires(Direction == SOAPY_SDR_RX && sizeof...(TBuffers) > 0UZ)
        [[maybe_unused]] int readStream(int& flags, long long& timeNs, std::uint32_t timeOutUs, TBuffers&&... ioBuffers) {
            if (_device.get() == nullptr || _stream.get() == nullptr) {
                return SOAPY_SDR_TIMEOUT;
            }
            if constexpr (sizeof...(TBuffers) == 1UZ) {
                auto&& buffer          = std::get<0>(std::tie(ioBuffers...));
                void*  ioBufferPointer = static_cast<void*>(buffer.data());
                return SoapySDRDevice_readStream(_device.get(), _stream.get(), &ioBufferPointer, buffer.size(), &flags, &timeNs, static_cast<long>(timeOutUs));
            } else {
                std::array<void*, sizeof...(ioBuffers)> ioBufferPointers = {static_cast<void*>(ioBuffers.data())...};
                return SoapySDRDevice_readStream(_device.get(), _stream.get(), ioBufferPointers.data(), std::get<0>(std::tie(ioBuffers...)).size(), &flags, &timeNs, timeOutUs);
            }
        }

        template<typename TCollection>
        requires(Direction == SOAPY_SDR_RX && requires(TCollection c) { c.begin()->data(); })
        [[maybe_unused]] int readStreamIntoBufferList(int& flags, long long& timeNs, long timeOutUs, TCollection&& ioBuffers) {
            std::vector<void*> ioBufferPointers;
            ioBufferPointers.reserve(ioBuffers.size());
            for (auto& ioBuffer : ioBuffers) {
                ioBufferPointers.emplace_back(ioBuffer.data());
            }
            return SoapySDRDevice_readStream(_device.get(), _stream.get(), ioBufferPointers.data(), ioBuffers.front().size(), &flags, &timeNs, timeOutUs);
        }

        // SoapySDR::Stream
        SoapySDRStream* get() const { return _stream.get(); }
    };

    std::vector<std::string> getStreamFormats(int direction, std::size_t channel) const {
        std::size_t length  = 0UZ;
        char**      formats = SoapySDRDevice_getStreamFormats(_device.get(), direction, channel, &length);
        return detail::convertToCpp(formats, length);
    }

    template<typename TValueType, int direction, typename TContainer = std::vector<std::size_t>>
    std::expected<Stream<TValueType, direction>, std::string> setupStream(const TContainer& channels = {0}) {
        if (!_device) {
            return std::unexpected(std::string("device not initialised"));
        }
        static_assert(std::is_unsigned_v<typename TContainer::value_type>, "channel indices must be unsigned integers");
        std::vector<std::size_t> channels_converted;
        std::transform(channels.begin(), channels.end(), std::back_inserter(channels_converted), [](auto val) { return static_cast<std::size_t>(val); });

        SoapySDRStream* stream = SoapySDRDevice_setupStream(_device.get(), direction, detail::toSoapySDRFormat<TValueType>(), channels_converted.data(), channels_converted.size(), nullptr);
        if (stream == nullptr) {
            return std::unexpected(std::format("setupStream failed: {}", SoapySDRDevice_lastError()));
        }
        return Stream<TValueType, direction>(_device, stream);
    }

    std::vector<ArgInfo> getSettingInfo() const {
        std::size_t      length = 0UZ;
        SoapySDRArgInfo* infos  = SoapySDRDevice_getSettingInfo(_device.get(), &length);
        return detail::convertToCpp(infos, length);
    }

    std::expected<void, std::string> writeSetting(const std::string& key, const std::string& value) {
        int ret = SoapySDRDevice_writeSetting(_device.get(), key.c_str(), value.c_str());
        if (ret != 0) {
            return std::unexpected(std::format("writeSetting({}, {}) error({}): {}", key, value, ret, SoapySDR_errToStr(ret)));
        }
        return {};
    }

    std::string readSetting(const std::string& key) const {
        char*       value = SoapySDRDevice_readSetting(_device.get(), key.c_str());
        std::string result(value);
        free(value);
        return result;
    }

    std::vector<ArgInfo> getChannelSettingInfo(int direction, std::size_t channel) const {
        std::size_t      length;
        SoapySDRArgInfo* infos = SoapySDRDevice_getChannelSettingInfo(_device.get(), direction, channel, &length);
        return detail::convertToCpp(infos, length);
    }

    std::expected<void, std::string> writeChannelSetting(int direction, std::size_t channel, const std::string& key, const std::string& value) {
        int ret = SoapySDRDevice_writeChannelSetting(_device.get(), direction, channel, key.c_str(), value.c_str());
        if (ret != 0) {
            return std::unexpected(std::format("writeChannelSetting({}, {}, {}, {}) error({}): {}", direction, channel, key, value, ret, SoapySDR_errToStr(ret)));
        }
        return {};
    }

    std::string readChannelSetting(int direction, std::size_t channel, const std::string& key) const {
        char*       value = SoapySDRDevice_readChannelSetting(_device.get(), direction, channel, key.c_str());
        std::string result(value);
        free(value);
        return result;
    }

    std::string getFrontendMapping(int direction) const {
        char*       mapping = SoapySDRDevice_getFrontendMapping(_device.get(), direction);
        std::string result(mapping);
        free(mapping);
        return result;
    }

    void setFrontendMapping(int direction, const std::string& mapping) { SoapySDRDevice_setFrontendMapping(_device.get(), direction, mapping.c_str()); }

    std::size_t getNumChannels(int direction) const { return SoapySDRDevice_getNumChannels(_device.get(), direction); }

    Kwargs getChannelInfo(int direction, std::size_t channel) const {
        SoapySDRKwargs kwargs = SoapySDRDevice_getChannelInfo(_device.get(), direction, channel);
        Kwargs         result;
        for (std::size_t i = 0; i < kwargs.size; ++i) {
            result[kwargs.keys[i]] = kwargs.vals[i];
        }
        SoapySDRKwargs_clear(&kwargs);
        return result;
    }

    bool getFullDuplex(int direction, std::size_t channel) const { return SoapySDRDevice_getFullDuplex(_device.get(), direction, channel); }
};
static_assert(std::is_default_constructible_v<Device>, "Device not default constructible");
static_assert(std::is_default_constructible_v<Device::Stream<float, SOAPY_SDR_RX>>, "Stream not default constructible");

} // namespace gr::blocks::sdr::soapy

template<>
struct std::formatter<SoapySDRRange> {
    constexpr auto parse(std::format_parse_context& ctx) const noexcept { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const gr::blocks::sdr::soapy::Range& range, FormatContext& ctx) const noexcept {
        return std::format_to(ctx.out(), "Range{{min: {}, max: {}, step: {}}}", range.minimum, range.maximum, range.step);
    }
};

#endif // SOAPYRAIIWRAPPER_HPP
