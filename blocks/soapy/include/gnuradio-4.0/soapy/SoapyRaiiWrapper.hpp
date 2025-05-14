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

#include <gnuradio-4.0/meta/formatter.hpp>

namespace gr::blocks::soapy {

using Range = SoapySDRRange;

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

/**
 * @brief RAII Wrapper for SoapySDRKwargs
 */
class KwargsWrapper {
    SoapySDRKwargs _cArgs;

public:
    KwargsWrapper(const SoapySDR::Kwargs& args) {
        std::memset(&_cArgs, 0, sizeof(_cArgs));
        for (const auto& it : args) {
            if (SoapySDRKwargs_set(&_cArgs, it.first.c_str(), it.second.c_str()) != 0) {
                throw std::bad_alloc();
            }
        }
    }

    ~KwargsWrapper() { SoapySDRKwargs_clear(&_cArgs); }

    [[nodiscard]] SoapySDRKwargs* get() noexcept { return &_cArgs; }
    operator const SoapySDRKwargs*() const noexcept { return &_cArgs; }
    operator SoapySDRKwargs*() noexcept { return &_cArgs; }
};

[[nodiscard]] inline SoapySDR::KwargsList convertToCpp(SoapySDRKwargs* results, std::size_t length) {
    SoapySDR::KwargsList kwargsList;
    kwargsList.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
        SoapySDR::Kwargs kwargs;
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
                assert(cInfo.options[j] != nullptr && "Option string should not be null");
                argInfo.options[j] = std::string(cInfo.options[j]);
            }
        }

        if (cInfo.optionNames != nullptr) {
            argInfo.optionNames.resize(cInfo.numOptions);
            for (std::size_t j = 0; j < cInfo.numOptions; ++j) {
                assert(cInfo.optionNames[j] != nullptr && "Option name string should not be null");
                argInfo.optionNames[j] = std::string(cInfo.optionNames[j]);
            }
        }
    }
    SoapySDRArgInfoList_clear(infos, size);

    return argInfoList;
}

// Mock function to simulate SoapySDRDevice_getSettingInfo
inline std::vector<ArgInfo> getMockDeviceSettingInfo() {
    // Simulating SoapySDRArgInfo array
    SoapySDRArgInfo* infos = new SoapySDRArgInfo[6];
    for (int i = 0; i < 6; ++i) {
        infos[i].key         = strdup("key");
        infos[i].value       = strdup("value");
        infos[i].name        = strdup("name");
        infos[i].description = strdup("description");
        infos[i].units       = strdup("units");
        infos[i].type        = SoapySDRArgInfoType::SOAPY_SDR_ARG_INFO_STRING;
        infos[i].range       = SoapySDRRange{0.0, 10.0, 1.0};
        infos[i].numOptions  = 3;
        infos[i].options     = new char*[3];
        infos[i].optionNames = new char*[3];
        for (int j = 0; j < 3; ++j) {
            infos[i].options[j]     = strdup("option");
            infos[i].optionNames[j] = strdup("optionName");
        }
    }
    return convertToCpp(infos, 6);
};

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
    return detail::convertToCpp(SoapySDR_listModules(&moduleCount), moduleCount);
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
    /**
     * Enumerate a list of available devices on the system.
     * \param args device construction key/value argument filters
     * \return a list of argument maps, each unique to a device
     */
    static SoapySDR::KwargsList enumerate(const SoapySDR::Kwargs& args = SoapySDR::Kwargs()) {
        std::size_t length = 0UZ;
        return detail::convertToCpp(SoapySDRDevice_enumerate(detail::KwargsWrapper(args), &length), length);
    }

    /**
     * Enumerate a list of available devices on the system.
     * Markup format for args: "keyA=valA, keyB=valB".
     * \param args a markup string of key/value argument filters
     * \return a list of argument maps, each unique to a device
     */
    static SoapySDR::KwargsList enumerate(const std::string& args) {
        SoapySDR::Kwargs  kwargs;
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

    /**
     * N.B. The device pointer will be internal to SoapySDR stored in a table
     * so subsequent calls with the same arguments will produce the same device.
     * This RAII wrapper ensures the requires matched call to unmake.
     *
     * \param args device construction key/value argument map
     */
    Device(const SoapySDR::Kwargs& args = SoapySDR::Kwargs(), std::source_location location = std::source_location::current()) {
        if (std::string_view(SOAPY_SDR_ABI_VERSION) != std::string_view(SoapySDR_getABIVersion())) {
            throw gr::exception(std::format("SoapySDR ABI mismatch: this {} vs. library {}", SOAPY_SDR_ABI_VERSION, SoapySDR_getABIVersion()), location);
        }

        _device.reset(SoapySDRDevice_make(detail::KwargsWrapper(args)), [location](SoapySDRDevice* device) {
            if (device == nullptr) {
                return;
            }
            if (int ret = SoapySDRDevice_unmake(device); ret) {
                throw gr::exception(std::format("error({}) closing Device: '{}'", ret, SoapySDR_errToStr(ret)), location);
            }
        });
        if (!_device) {
            throw gr::exception(std::format("Device({}) - SoapySDRDevice_make failed", args), location);
        }
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

    /*******************************************************************
     * Antenna API
     ******************************************************************/

    std::vector<std::string> listAvailableAntennas(int direction, std::size_t channel) const {
        std::size_t length = 0UZ;
        return detail::convertToCpp(SoapySDRDevice_listAntennas(_device.get(), direction, channel, &length), length);
    }

    void setAntenna(int direction, std::size_t channel, std::string_view antennaName, std::source_location location = std::source_location::current()) {
        if (int error = SoapySDRDevice_setAntenna(_device.get(), direction, channel, antennaName.data()); error) {
            throw gr::exception(std::format("Soapy error ({}): {}", error, SoapySDR_errToStr(error)), location);
        }
    }

    std::string getAntenna(int direction, std::size_t channel) const {
        char*       value = SoapySDRDevice_getAntenna(_device.get(), direction, channel);
        std::string result(value);
        free(value);
        return result;
    }

    /*******************************************************************
     * Gain API
     ******************************************************************/

    std::vector<std::string> listAvailableGainElements(int direction, std::size_t channel) const {
        std::size_t length = 0UZ;
        return detail::convertToCpp(SoapySDRDevice_listGains(_device.get(), direction, channel, &length), length);
    }

    bool hasAutomaticGainControl(int direction, std::size_t channel) const { return SoapySDRDevice_hasGainMode(_device.get(), direction, channel); }

    void setAutomaticGainControl(int direction, std::size_t channel, bool automatic, std::source_location location = std::source_location::current()) {
        if (int error = SoapySDRDevice_setGainMode(_device.get(), direction, channel, automatic); error) {
            throw gr::exception(std::format("setGain({}, {}, {}) - Soapy error ({}): {}", direction, channel, automatic, error, SoapySDR_errToStr(error), location));
        }
    }

    bool isAutomaticGainControl(int direction, std::size_t channel) { return SoapySDRDevice_getGainMode(_device.get(), direction, channel); }

    void setGain(int direction, std::size_t channel, double gain_dB, std::string_view gainElement = {}, std::source_location location = std::source_location::current()) {
        if (int error = (gainElement.empty() ? SoapySDRDevice_setGain(_device.get(), direction, channel, gain_dB) //
                                             : SoapySDRDevice_setGainElement(_device.get(), direction, channel, gainElement.data(), gain_dB));
            error) {
            throw gr::exception(std::format("setGain({}, {}, {}) - Soapy error ({}): {}", direction, channel, gain_dB, error, SoapySDR_errToStr(error), location));
        }
    }

    double getGain(int direction, std::size_t channel, std::string_view gainElement = {}) const {
        if (gainElement.empty()) {
            return SoapySDRDevice_getGain(_device.get(), direction, channel);
        }
        return SoapySDRDevice_getGainElement(_device.get(), direction, channel, gainElement.data());
    }

    /*******************************************************************
     * Bandwidth API
     ******************************************************************/

    std::vector<double> listAvailableBandwidths(int direction, std::size_t channel) const {
        std::size_t         length     = 0UZ;
        double*             bandwidths = SoapySDRDevice_listBandwidths(_device.get(), direction, channel, &length);
        std::vector<double> bwList(bandwidths, bandwidths + length);
        free(bandwidths);
        return bwList;
    }

    void setBandwidth(int direction, std::size_t channel, double bandwidthHz, std::source_location location = std::source_location::current()) {
        if (int error = SoapySDRDevice_setBandwidth(_device.get(), direction, channel, bandwidthHz); error) {
            throw gr::exception(std::format("setBandwidth({}, {}, {}) - Soapy error ({}): {}", direction, channel, bandwidthHz, error, SoapySDR_errToStr(error)), location);
        }
    }

    double getBandwidth(int direction, std::size_t channel) const { return SoapySDRDevice_getBandwidth(_device.get(), direction, channel); }

    /*******************************************************************
     * Frequency API
     ******************************************************************/

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

    /**
     * Set the center frequency of the chain.
     *  - For RX, this specifies the down-conversion frequency.
     *  - For TX, this specifies the up-conversion frequency.
     *
     * The default implementation of setFrequency() will tune the "RF"
     * component as close as possible to the requested center frequency.
     * Tuning inaccuracies will be compensated for with the "BB" component.
     *
     * The args can be used to augment the tuning algorithm.
     *  - Use "OFFSET" to specify an "RF" tuning offset,
     *    usually with the intention of moving the LO out of the passband.
     *    The offset will be compensated for using the "BB" component.
     *  - Use the name of a component for the key and a frequency in Hz
     *    as the value (any format) to enforce a specific frequency.
     *    The other components will be tuned with compensation
     *    to achieve the specified overall frequency.
     *  - Use the name of a component for the key and the value "IGNORE"
     *    so that the tuning algorithm will avoid altering the component.
     *  - Vendor specific implementations can also use the same args to augment
     *    tuning in other ways such as specifying fractional vs integer N tuning.
     *
     * \param device a pointer to a device instance
     * \param direction the channel direction RX or TX
     * \param channel an available channel on the device
     * \param frequency the center frequency in Hz
     * \param args optional tuner arguments
     * \page location optional source location
     */
    void setCenterFrequency(int direction, std::size_t channel, double frequency, const SoapySDR::Kwargs& args = SoapySDR::Kwargs(), std::source_location location = std::source_location::current()) {
        if (int error = SoapySDRDevice_setFrequency(_device.get(), direction, channel, frequency, detail::KwargsWrapper(args)); error) {
            throw gr::exception(std::format("setFrequency({}, {}, {}, {}) - Soapy error ({}): {}", direction, channel, frequency, args, error, SoapySDR_errToStr(error)), location);
        }
    }

    double getCenterFrequency(int direction, std::size_t channel) const { return SoapySDRDevice_getFrequency(_device.get(), direction, channel); }

    /*******************************************************************
     * Sample Rate API
     ******************************************************************/

    std::vector<double> listSampleRates(int direction, std::size_t channel) const {
        std::size_t         length = 0UZ;
        double*             rates  = SoapySDRDevice_listSampleRates(_device.get(), direction, channel, &length);
        std::vector<double> rateList(rates, rates + length);
        free(rates);
        return rateList;
    }

    void setSampleRate(int direction, std::size_t channel, double rate, std::source_location location = std::source_location::current()) {
        if (int error = SoapySDRDevice_setSampleRate(_device.get(), direction, channel, rate); error) {
            throw gr::exception(std::format("setSampleRate({}, {}, {}) - Soapy error ({}): {}", direction, channel, rate, error, SoapySDR_errToStr(error)), location);
        }
    }

    double getSampleRate(int direction, std::size_t channel) const { return SoapySDRDevice_getSampleRate(_device.get(), direction, channel); }

    /*******************************************************************
     * Time API
     ******************************************************************/

    std::vector<std::string> listAvailableTimeSources() const {
        std::size_t length = 0UZ;
        return detail::convertToCpp(SoapySDRDevice_listTimeSources(_device.get(), &length), length);
    }

    std::string getTimeSource() const {
        char*       value = SoapySDRDevice_getTimeSource(_device.get());
        std::string result(value);
        free(value);
        return result;
    }

    void setTimeSource(const std::string& source, std::source_location location = std::source_location::current()) {
        if (int error = SoapySDRDevice_setTimeSource(_device.get(), source.c_str()); error) {
            throw gr::exception(std::format("Soapy error ({}): {}", error, SoapySDR_errToStr(error)), location);
        }
    }

    std::uint64_t getHardwareTime(std::string_view what = {}) const { return static_cast<std::uint64_t>(SoapySDRDevice_getHardwareTime(_device.get(), what.data())); }

    void setHardwareTime(long long timeNs, const std::string& event = "", std::source_location location = std::source_location::current()) {
        if (int error = SoapySDRDevice_setHardwareTime(_device.get(), timeNs, event.empty() ? nullptr : event.c_str()); error) {
            throw gr::exception(std::format("Soapy error ({}): {}", error, SoapySDR_errToStr(error)), location);
        }
    }

    /**
     * Get the master clock rate of the device.
     * \return the clock rate in Hz
     */
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
                      throw gr::exception(std::format("error({}) closing Device: '{}'", ret, SoapySDR_errToStr(ret)));
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

        /**
         * Activate a stream.
         * Call activate to prepare a stream before using read/write().
         * The implementation control switches or stimulate data flow.
         *
         * The timeNs is only valid when the flags have SOAPY_SDR_HAS_TIME.
         * The numElems count can be used to request a finite burst size.
         * The SOAPY_SDR_END_BURST flag can signal end on the finite burst.
         * Not all implementations will support the full range of options.
         * In this case, the implementation returns SOAPY_SDR_NOT_SUPPORTED.
         *
         * \param flags optional flag indicators about the stream
         * \param timeNs optional activation time in nanoseconds
         * \param numElems optional element count for burst control
         */
        [[maybe_unused]] int activate(int flags = 0, long long timeNs = 0, std::size_t numElems = 0UZ) {
            int ret = SoapySDRDevice_activateStream(_device.get(), _stream.get(), flags, timeNs, numElems);
            if (ret != 0) {
                throw gr::exception(std::format("failed to activate(flags= {}, timeNs= {}, numElems= {}), stream - error({}) - '{}'", //
                    flags, timeNs, numElems, ret, SoapySDR_errToStr(ret)));
            }
            return ret;
        }

        /**
         * Deactivate a stream.
         * Call deactivate when not using using read/write().
         * The implementation control switches or halt data flow.
         *
         * The timeNs is only valid when the flags have SOAPY_SDR_HAS_TIME.
         * Not all implementations will support the full range of options.
         * In this case, the implementation returns SOAPY_SDR_NOT_SUPPORTED.
         *
         * \param flags optional flag indicators about the stream
         * \param timeNs optional deactivation time in nanoseconds
         * \return 0 for success or error code on failure
         */
        [[maybe_unused]] int deactivate(int flags = 0, long long timeNs = 0) {
            int ret = SoapySDRDevice_deactivateStream(_device.get(), _stream.get(), flags, timeNs);
            if (ret != 0) {
                throw gr::exception(std::format("failed to deactivate(flags= {}, timeNs= {}), stream - error({}) - '{}'", //
                    flags, timeNs, ret, SoapySDR_errToStr(ret)));
            }
            return ret;
        }

        /*!
         * Read elements from a stream for reception.
         * This is a multi-channel call, and buffs should be an array of void *,
         * where each pointer will be filled with data from a different channel.
         *
         * **Client code compatibility:**
         * The readStream() call should be well defined at all times,
         * including prior to activation and after deactivation.
         * When inactive, readStream() should implement the timeout
         * specified by the caller and return SOAPY_SDR_TIMEOUT.
         *
         * \param ioBuffers a collection of collection<TValueType> cast (that is eventually cast to void* buffers x num chans)
         * \param [out] flags optional flag indicators about the result
         * \param [out] timeNs the buffer's timestamp in nanoseconds
         * \param timeOutUs the timeout in microseconds
         * \return the number of elements read per buffer or error code
         */
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

        /**
         * Read elements from a stream for reception.
         * This is a multi-channel call, and buffs should be an array of void *,
         * where each pointer will be filled with data from a different channel.
         *
         * **Client code compatibility:**
         * The readStream() call should be well defined at all times,
         * including prior to activation and after deactivation.
         * When inactive, readStream() should implement the timeout
         * specified by the caller and return SOAPY_SDR_TIMEOUT.
         *
         * \param [out] flags optional flag indicators about the result
         * \param [out] timeNs the buffer's timestamp in nanoseconds
         * \param timeOutUs the timeout in microseconds
         * \param ioBuffers a collection of collection<TValueType> cast (that is eventually cast to void* buffers x num chans)
         * \return the number of elements read per buffer or error code
         */
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

    /**
     * Initialize a stream given a list of channels and stream arguments.
     * The implementation may change switches or power-up components.
     * All stream API calls should be usable with the new stream object
     * after setupStream() is complete, regardless of the activity state.
     *
     * The API allows any number of simultaneous TX and RX streams, but many dual-channel
     * devices are limited to one stream in each direction, using either one or both channels.
     * This call will return an error if an unsupported combination is requested,
     * or if a requested channel in this direction is already in use by another stream.
     *
     * When multiple channels are added to a stream, they are typically expected to have
     * the same sample rate. See SoapySDRDevice_setSampleRate().
     *
     * \param device a pointer to a device instance
     * \return the opaque pointer to a stream handle.
     * \parblock
     *
     * The returned stream is not required to have internal locking, and may not be used
     * concurrently from multiple threads.
     * \endparblock
     *
     * \param direction the channel direction (`SOAPY_SDR_RX` or `SOAPY_SDR_TX`)
     * \param format A string representing the desired buffer format in read/writeStream()
     * \parblock
     *
     * The first character selects the number type:
     *   - "C" means complex
     *   - "F" means floating point
     *   - "S" means signed integer
     *   - "U" means unsigned integer
     *
     * The type character is followed by the number of bits per number (complex is 2x this size per sample)
     *
     *  Example format strings:
     *   - "CF32" -  complex float32 (8 bytes per element)
     *   - "CS16" -  complex int16 (4 bytes per element)
     *   - "CS12" -  complex int12 (3 bytes per element)
     *   - "CS4" -  complex int4 (1 byte per element)
     *   - "S32" -  int32 (4 bytes per element)
     *   - "U8" -  uint8 (1 byte per element)
     *
     * \endparblock
     * \param channels a list of channels or empty for automatic
     * \param numChans the number of elements in the channels array
     * \param args stream args or empty for defaults
     * \parblock
     *
     *   Recommended keys to use in the args dictionary:
     *    - "WIRE" - format of the samples between device and host
     * \endparblock
     * \return the stream pointer or nullptr for failure
     */
    template<typename TValueType, int direction, typename TContainer = std::vector<std::size_t>>
    Stream<TValueType, direction> setupStream(const TContainer& channels = {0}) {
        assert(_device.get() != nullptr);
        static_assert(std::is_unsigned_v<typename TContainer::value_type>, "channel indices must be unsigned integers");
        std::vector<std::size_t> channels_converted; // convert to std::vector<size_t>
        std::transform(channels.begin(), channels.end(), std::back_inserter(channels_converted), [](auto val) { return static_cast<std::size_t>(val); });

        SoapySDRStream* stream = SoapySDRDevice_setupStream(_device.get(), direction, detail::toSoapySDRFormat<TValueType>(), channels_converted.data(), channels_converted.size(), nullptr);
        if (stream == nullptr) {
            throw gr::exception(std::format("failed to setup Stream<{},{}>([{}]) : {}", meta::type_name<TValueType>(), direction, gr::join(channels, ", "), SoapySDRDevice_lastError()));
        }
        return Stream<TValueType, direction>(_device, stream);
    }

    /**
     * Describe the allowed keys and values used for settings.
     * \return a list of argument info structures
     */
    std::vector<ArgInfo> getSettingInfo() const {
        std::size_t length = 0UZ;
        return detail::convertToCpp(SoapySDRDevice_getSettingInfo(_device.get(), &length), length);
    }

    /**
     * Write an arbitrary setting on the device.
     * \param key the setting identifier
     * \param value the setting value
     */
    void writeSetting(const std::string& key, const std::string& value) {
        int ret = SoapySDRDevice_writeSetting(_device.get(), key.c_str(), value.c_str());
        if (ret != 0) {
            throw gr::exception(std::format("Error writing setting ({}): '{}'", ret, SoapySDR_errToStr(ret)));
        }
    }

    /**
     * Read an arbitrary setting on the device.
     * \param key the setting identifier
     * \return the setting value
     */
    std::string readSetting(const std::string& key) const {
        char*       value = SoapySDRDevice_readSetting(_device.get(), key.c_str());
        std::string result(value);
        free(value);
        return result;
    }

    /*******************************************************************
     * Channels API
     ******************************************************************/

    std::vector<ArgInfo> getChannelSettingInfo(int direction, std::size_t channel) const {
        std::size_t      length;
        SoapySDRArgInfo* infos = SoapySDRDevice_getChannelSettingInfo(_device.get(), direction, channel, &length);
        return detail::convertToCpp(infos, length);
    }

    void writeChannelSetting(int direction, std::size_t channel, const std::string& key, const std::string& value) {
        int ret = SoapySDRDevice_writeChannelSetting(_device.get(), direction, channel, key.c_str(), value.c_str());
        if (ret != 0) {
            throw gr::exception(std::format("Error writing channel setting ({}): '{}'", ret, SoapySDR_errToStr(ret)));
        }
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

    SoapySDR::Kwargs getChannelInfo(int direction, std::size_t channel) const {
        SoapySDRKwargs   kwargs = SoapySDRDevice_getChannelInfo(_device.get(), direction, channel);
        SoapySDR::Kwargs result;
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

} // namespace gr::blocks::soapy

template<>
struct std::formatter<SoapySDRRange> {
    constexpr auto parse(std::format_parse_context& ctx) const noexcept { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const gr::blocks::soapy::Range& range, FormatContext& ctx) const noexcept {
        return std::format_to(ctx.out(), "Range{{min: {}, max: {}, step: {}}}", range.minimum, range.maximum, range.step);
    }
};

#endif // SOAPYRAIIWRAPPER_HPP
