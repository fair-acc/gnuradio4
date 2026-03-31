#ifndef GNURADIO_AUDIO_BACKENDS_HPP
#define GNURADIO_AUDIO_BACKENDS_HPP

#include <gnuradio-4.0/CircularBuffer.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace gr::audio::detail {

template<typename T>
concept AudioSample = std::same_as<T, float> || std::same_as<T, std::int16_t>;

struct AudioDeviceConfig {
    std::uint32_t sampleRate{0U};
    std::uint32_t numChannels{0U};
    std::size_t   bufferFrames{0U};
    std::string   device;
    bool          useDummyBackendForTests{false};
};

struct AudioStreamFormat {
    std::uint32_t sampleRate{0U};
    std::uint32_t numChannels{0U};
};

struct AudioDeviceInfo {
    std::string name;
    std::string id;
};

[[nodiscard]] inline std::uint64_t wallClockNs() { return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()); }

[[nodiscard]] inline bool caseInsensitiveContains(std::string_view haystack, std::string_view needle) {
    if (needle.empty() || needle.size() > haystack.size()) {
        return needle.empty();
    }
    return std::ranges::search(haystack, needle, [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b)); }).begin() != haystack.end();
}

[[nodiscard]] inline bool isDefaultDevice(std::string_view spec) {
    if (spec.empty()) {
        return true;
    }
    return caseInsensitiveContains("default", spec) && spec.size() <= 7U;
}

[[nodiscard]] inline std::optional<std::size_t> resolveDeviceIndex(std::string_view deviceSpec, std::span<const AudioDeviceInfo> devices) {
    if (isDefaultDevice(deviceSpec)) {
        return std::nullopt; // caller uses system default
    }

    constexpr std::string_view kIdPrefix = "@id:";
    if (deviceSpec.starts_with(kIdPrefix)) {
        const auto targetId = deviceSpec.substr(kIdPrefix.size());
        for (std::size_t i = 0U; i < devices.size(); ++i) {
            if (devices[i].id == targetId) {
                return i;
            }
        }
        return std::nullopt;
    }

    for (std::size_t i = 0U; i < devices.size(); ++i) {
        if (caseInsensitiveContains(devices[i].name, deviceSpec)) {
            return i;
        }
    }
    return std::nullopt;
}

[[nodiscard]] inline std::vector<std::string> formatDeviceList(std::span<const AudioDeviceInfo> devices) {
    std::vector<std::string> result;
    result.reserve(devices.size());
    for (const auto& dev : devices) {
        result.push_back(std::format("{} [{}]", dev.name, dev.id));
    }
    return result;
}

template<AudioSample T>
struct AudioStateBase {
    using SampleBuffer = gr::CircularBuffer<T, std::dynamic_extent, gr::ProducerType::Single>;
    using SampleWriter = decltype(std::declval<SampleBuffer&>().new_writer());
    using SampleReader = decltype(std::declval<SampleBuffer&>().new_reader());

    std::atomic<bool>        stopRequested{false};
    std::atomic<std::size_t> overflowCount{0U};
    std::atomic<std::size_t> underrunCount{0U};
    SampleBuffer             buffer{1U};
    SampleWriter             writer{buffer.new_writer()};
    SampleReader             reader{buffer.new_reader()};

    [[nodiscard]] static std::size_t bufferCapacitySamples(std::size_t numChannels, std::size_t bufferFrames) { return std::max<std::size_t>(1U, numChannels) * std::max<std::size_t>(1U, bufferFrames); }

    void recreateBuffer(std::size_t capacitySamples) {
        buffer = SampleBuffer(std::max<std::size_t>(1U, capacitySamples));
        writer = buffer.new_writer();
        reader = buffer.new_reader();
        overflowCount.store(0U, std::memory_order_relaxed);
        underrunCount.store(0U, std::memory_order_relaxed);
    }
};

[[nodiscard]] inline std::size_t wholeFrameSamples(std::size_t sampleCount, std::size_t channelCount) {
    const std::size_t alignedChannelCount = std::max<std::size_t>(1U, channelCount);
    return sampleCount - (sampleCount % alignedChannelCount);
}

template<AudioSample T>
struct AudioSinkState : AudioStateBase<T> {
    using AudioStateBase<T>::reader;
    using AudioStateBase<T>::stopRequested;
    using AudioStateBase<T>::writer;

    template<typename InputSpan>
    [[nodiscard]] std::size_t writeFromInput(const InputSpan& inSpan, std::size_t channelCount) {
        if (stopRequested.load(std::memory_order_acquire)) {
            return 0U;
        }

        const std::size_t alignedChannelCount = std::max<std::size_t>(1U, channelCount);
        const std::size_t nFrameSamples       = wholeFrameSamples(inSpan.size(), alignedChannelCount);
        if (nFrameSamples == 0U) {
            return 0U;
        }

        std::size_t available = writer.available();
        if (available == 0U) {
            return 0U;
        }
        available = wholeFrameSamples(available, alignedChannelCount);
        if (available == 0U) {
            return 0U;
        }

        const std::size_t nChunk     = std::min(nFrameSamples, available);
        auto              writerSpan = writer.tryReserve(nChunk);
        if (writerSpan.empty()) {
            return 0U;
        }

        const std::size_t published = wholeFrameSamples(writerSpan.size(), alignedChannelCount);
        if (published == 0U) {
            return 0U;
        }

        std::copy_n(inSpan.begin(), static_cast<std::ptrdiff_t>(published), writerSpan.begin());
        writerSpan.publish(published);
        return published;
    }

    void readPlanarFloat(float* output, std::size_t frameCount, std::size_t channelCount) {
        if (output == nullptr || frameCount == 0U) {
            return;
        }

        const auto toFloatSample = [](T value) {
            if constexpr (std::same_as<T, float>) {
                return value;
            } else {
                constexpr float kScale = 1.0f / 32768.0f;
                return static_cast<float>(value) * kScale;
            }
        };

        const std::size_t alignedChannelCount = std::max<std::size_t>(1U, channelCount);
        const std::size_t available           = reader.available();
        const std::size_t alignedAvailable    = wholeFrameSamples(available, alignedChannelCount);
        const std::size_t sampleCount         = frameCount * alignedChannelCount;
        const std::size_t nRead               = std::min(sampleCount, alignedAvailable);
        const std::size_t readFrames          = nRead / alignedChannelCount;

        if (nRead > 0U) {
            auto read = reader.get(nRead);
            for (std::size_t frame = 0U; frame < readFrames; ++frame) {
                for (std::size_t channel = 0U; channel < alignedChannelCount; ++channel) {
                    output[channel * frameCount + frame] = toFloatSample(read[frame * alignedChannelCount + channel]);
                }
            }
            std::ignore = read.consume(nRead);
        }

        if (readFrames < frameCount) {
            this->underrunCount.fetch_add(1U, std::memory_order_relaxed);
        }
        for (std::size_t channel = 0U; channel < alignedChannelCount; ++channel) {
            std::fill_n(output + static_cast<std::ptrdiff_t>(channel * frameCount + readFrames), static_cast<std::ptrdiff_t>(frameCount - readFrames), 0.0f);
        }
    }
};

template<AudioSample T>
struct AudioSourceState : AudioStateBase<T> {
    using AudioStateBase<T>::reader;
    using AudioStateBase<T>::stopRequested;
    using AudioStateBase<T>::writer;

    [[nodiscard]] std::size_t writePlanarFloat(const float* input, std::size_t frameCount, std::size_t inputChannels, std::size_t outputChannels) {
        if (stopRequested.load(std::memory_order_acquire) || input == nullptr || frameCount == 0U) {
            return 0U;
        }

        const auto fromFloatSample = [](float value) {
            if constexpr (std::same_as<T, float>) {
                return value;
            } else {
                const float clamped = std::clamp(value, -1.0f, 1.0f);
                if (clamped <= -1.0f) {
                    return std::numeric_limits<std::int16_t>::min();
                }
                const auto scaled = std::lround(static_cast<double>(clamped) * static_cast<double>(std::numeric_limits<std::int16_t>::max()));
                return static_cast<std::int16_t>(std::clamp<long>(static_cast<long>(scaled), std::numeric_limits<std::int16_t>::min(), std::numeric_limits<std::int16_t>::max()));
            }
        };

        const std::size_t alignedOutputChannels = std::max<std::size_t>(1U, outputChannels);
        const std::size_t alignedAvailable      = wholeFrameSamples(writer.available(), alignedOutputChannels);
        const std::size_t requested             = frameCount * alignedOutputChannels;
        const std::size_t nSamplesToWrite       = std::min(requested, alignedAvailable);
        if (nSamplesToWrite == 0U) {
            if (requested > 0U) {
                this->overflowCount.fetch_add(1U, std::memory_order_relaxed);
            }
            return 0U;
        }

        auto writeSpan = writer.tryReserve(nSamplesToWrite);
        if (writeSpan.empty()) {
            return 0U;
        }

        const std::size_t published   = wholeFrameSamples(writeSpan.size(), alignedOutputChannels);
        const std::size_t chunkFrames = published / alignedOutputChannels;
        for (std::size_t frame = 0U; frame < chunkFrames; ++frame) {
            for (std::size_t channel = 0U; channel < alignedOutputChannels; ++channel) {
                const float value                                  = channel < inputChannels ? input[channel * frameCount + frame] : 0.0f;
                writeSpan[frame * alignedOutputChannels + channel] = fromFloatSample(value);
            }
        }

        writeSpan.publish(published);
        return published;
    }

    [[nodiscard]] std::size_t readToOutput(std::span<T> output, std::size_t channelCount) {
        const std::size_t alignedOutputSize = wholeFrameSamples(output.size(), channelCount);
        const std::size_t alignedAvailable  = wholeFrameSamples(reader.available(), channelCount);
        const std::size_t nSamplesToRead    = std::min(alignedOutputSize, alignedAvailable);
        if (nSamplesToRead == 0U) {
            return 0U;
        }

        auto readSpan = reader.get(nSamplesToRead);
        if (readSpan.empty()) {
            return 0U;
        }

        std::copy_n(readSpan.begin(), static_cast<std::ptrdiff_t>(readSpan.size()), output.begin());
        std::ignore = readSpan.consume(readSpan.size());
        return readSpan.size();
    }
};

} // namespace gr::audio::detail

#endif // GNURADIO_AUDIO_BACKENDS_HPP
