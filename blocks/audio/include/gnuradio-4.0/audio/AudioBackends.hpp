#ifndef GNURADIO_AUDIO_BACKENDS_HPP
#define GNURADIO_AUDIO_BACKENDS_HPP

#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/Message.hpp>

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
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace gr::audio {

enum class AudioPortMode { interleaved, multiplexed };
enum class ChannelPadding { zero, cyclic };

namespace detail {

inline constexpr std::size_t kMaxAudioChannels          = 32UZ;
inline constexpr std::size_t kMaxBatchFrames            = 8192UZ;
inline constexpr std::size_t kMinBufferFrames           = 8192UZ;
inline constexpr double      kMaxServoRatioDeviation    = 0.001;
inline constexpr double      kPlaybackFillTargetSeconds = 0.1;

enum class AudioBackendState : int { pending, running, suspended, denied, unavailable, failed, stopped, ended, muted };

template<typename T>
concept AudioSample = std::same_as<T, float> || std::same_as<T, std::int16_t>;

struct AudioDeviceConfig {
    std::uint32_t  sampleRate{0U};
    std::uint32_t  numChannels{0U};
    float          bufferSeconds{0.f};
    std::string    device{};
    bool           useDummyBackendForTests{false};
    ChannelPadding channelPadding{ChannelPadding::zero};
};

struct AudioStreamFormat {
    std::uint32_t sampleRate{0U};
    std::uint32_t numChannels{0U};
};

struct AudioDeviceInfo {
    std::string name;
    std::string id;
};

[[nodiscard]] inline std::uint32_t validSampleRate(float value) {
    if (!std::isfinite(value) || value < 1.f || static_cast<double>(value) > static_cast<double>(std::numeric_limits<int>::max())) {
        return 0U;
    }
    return static_cast<std::uint32_t>(std::lround(static_cast<double>(value)));
}

[[nodiscard]] inline std::uint32_t validChannelCount(std::size_t channels) { return channels >= 1UZ && channels <= kMaxAudioChannels ? static_cast<std::uint32_t>(channels) : 0U; }

[[nodiscard]] inline std::size_t bufferFramesFor(float bufferSeconds, std::uint32_t sampleRate) { return std::max(kMinBufferFrames, static_cast<std::size_t>(static_cast<double>(std::max(0.f, bufferSeconds)) * sampleRate)); }

[[nodiscard]] inline std::optional<std::size_t> paddingSourceChannel(std::size_t channel, std::size_t inputChannels, ChannelPadding padding) {
    if (channel < inputChannels) {
        return channel;
    }
    if (padding == ChannelPadding::cyclic && inputChannels != 0UZ) {
        return channel % inputChannels;
    }
    return std::nullopt;
}

template<AudioSample T, typename TRead>
void adaptChannels(std::span<T> output, std::size_t frames, std::size_t inputChannels, std::size_t outputChannels, ChannelPadding padding, TRead&& readSample) {
    for (std::size_t channel = 0UZ; channel < outputChannels; ++channel) {
        const std::optional<std::size_t> source = paddingSourceChannel(channel, inputChannels, padding);
        for (std::size_t frame = 0UZ; frame < frames; ++frame) {
            output[frame * outputChannels + channel] = source ? readSample(frame, *source) : T{};
        }
    }
}

[[nodiscard]] inline std::size_t playbackTransferSamples(std::size_t available, std::size_t backendSpace, std::size_t scratchSpace, std::size_t channels) {
    if (channels == 0UZ) {
        return 0UZ;
    }
    const std::size_t outputFrames = std::min(backendSpace, scratchSpace) / channels;
    const std::size_t inputFrames  = outputFrames > 1UZ ? static_cast<std::size_t>(static_cast<double>(outputFrames - 1UZ) * (1.0 - kMaxServoRatioDeviation)) : 0UZ;
    return std::min(available / channels, inputFrames) * channels;
}

class AudioDiagnostics {
    mutable std::mutex _mutex;
    gr::property_map   _snapshot{{"state", std::string("stopped")}, {"sample_rate", 0.f}};

public:
    [[nodiscard]] gr::property_map get() const {
        std::lock_guard lock(_mutex);
        return _snapshot;
    }

    void update(std::invocable<gr::property_map&> auto&& modify) {
        std::lock_guard lock(_mutex);
        modify(_snapshot);
    }

    void refreshCounters(const auto& state) {
        const auto overflow = static_cast<std::uint64_t>(state.overflowCount.load(std::memory_order_relaxed));
        const auto underrun = static_cast<std::uint64_t>(state.underrunCount.load(std::memory_order_relaxed));
        update([&](gr::property_map& snapshot) {
            snapshot.insert_or_assign("overflow_count", overflow);
            snapshot.insert_or_assign("underrun_count", underrun);
        });
    }

    void markStopped() {
        update([](gr::property_map& snapshot) {
            snapshot.insert_or_assign("last_stream_state", snapshot.value_or<std::string>("state", "stopped"));
            snapshot.insert_or_assign("state", std::string("stopped"));
        });
    }

    [[nodiscard]] std::optional<gr::Message> reply(gr::Message message) const {
        if (message.cmd != gr::message::Command::Get) {
            message.data = std::unexpected(gr::Error("audio_backend is read-only"));
        } else {
            message.data = get();
        }
        return message;
    }
};

[[nodiscard]] inline std::uint64_t wallClockNs() { return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()); }

inline constexpr auto equalIgnoringCase = [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b)); };

[[nodiscard]] inline bool isDefaultDevice(std::string_view spec) { return spec.empty() || std::ranges::equal(spec, std::string_view("default"), equalIgnoringCase); }

[[nodiscard]] inline std::optional<std::size_t> resolveDeviceIndex(std::string_view deviceSpec, std::span<const AudioDeviceInfo> devices) {
    if (isDefaultDevice(deviceSpec)) {
        return std::nullopt;
    }
    constexpr std::string_view kIdPrefix        = "@id:";
    const auto                 nameContainsSpec = [deviceSpec](const AudioDeviceInfo& candidate) { return !std::ranges::search(candidate.name, deviceSpec, equalIgnoringCase).empty(); };
    const auto                 match            = deviceSpec.starts_with(kIdPrefix) ? std::ranges::find(devices, deviceSpec.substr(kIdPrefix.size()), &AudioDeviceInfo::id) : std::ranges::find_if(devices, nameContainsSpec);
    return match == devices.end() ? std::nullopt : std::optional(static_cast<std::size_t>(std::distance(devices.begin(), match)));
}

[[nodiscard]] inline std::vector<std::string> formatDeviceList(std::span<const AudioDeviceInfo> devices) {
    return devices | std::views::transform([](const AudioDeviceInfo& device) { return std::format("{} [{}]", device.name, device.id); }) | std::ranges::to<std::vector>();
}

template<AudioSample T>
struct AudioStateBase {
    using SampleBuffer = gr::CircularBuffer<T, std::dynamic_extent, gr::ProducerType::Single>;
    using SampleWriter = decltype(std::declval<SampleBuffer&>().new_writer());
    using SampleReader = decltype(std::declval<SampleBuffer&>().new_reader());

    std::atomic<bool>           stopRequested{false};
    std::atomic<std::size_t>    overflowCount{0U};
    std::atomic<std::size_t>    underrunCount{0U};
    std::atomic<std::size_t>    observedChannels{0U};
    std::atomic<ChannelPadding> channelPadding{ChannelPadding::zero};
    std::atomic<std::size_t>    numChannels{0UZ};
    SampleBuffer                buffer{1U};
    SampleWriter                writer{buffer.new_writer()};
    SampleReader                reader{buffer.new_reader()};

    [[nodiscard]] static std::size_t bufferCapacitySamples(std::size_t numChannels, std::size_t bufferFrames) { return std::max<std::size_t>(1U, numChannels) * std::max<std::size_t>(1U, bufferFrames); }

    void recreateBuffer(std::size_t capacitySamples) {
        buffer = SampleBuffer(std::max<std::size_t>(1U, capacitySamples));
        writer = buffer.new_writer();
        reader = buffer.new_reader();
        overflowCount.store(0U, std::memory_order_relaxed);
        underrunCount.store(0U, std::memory_order_relaxed);
        observedChannels.store(0UZ, std::memory_order_relaxed);
    }
};

[[nodiscard]] inline std::size_t wholeFrameSamples(std::size_t sampleCount, std::size_t channelCount) {
    const std::size_t alignedChannelCount = std::max<std::size_t>(1U, channelCount);
    return sampleCount - (sampleCount % alignedChannelCount);
}

/// Single-producer/single-consumer ring of planar float blocks shared with a JavaScript AudioWorkletProcessor through
/// WebAssembly memory. The worklet indexes `samples`, `blockFrames` and `blockChannels` by slot and moves `head`/`tail`
/// with Atomics; `missedBlocks` counts the blocks it could not deliver (capture, ring full) or fetch (playback, empty).
struct WorkletBlockRing {
    static constexpr std::size_t kBlockFrames = 128UZ;

    std::atomic<std::uint32_t> head{0U};
    std::atomic<std::uint32_t> tail{0U};
    std::atomic<std::uint32_t> missedBlocks{0U};
    std::uint32_t              blockCount;
    std::uint32_t              channelStride;
    std::vector<std::uint32_t> blockFrames;
    std::vector<std::uint32_t> blockChannels;
    std::vector<float>         samples;

    WorkletBlockRing(std::size_t nBlocks, std::size_t nChannels) : blockCount(static_cast<std::uint32_t>(std::max(1UZ, nBlocks))), channelStride(static_cast<std::uint32_t>(std::max(1UZ, nChannels))), blockFrames(blockCount), blockChannels(blockCount), samples(static_cast<std::size_t>(blockCount) * channelStride * kBlockFrames) {}

    [[nodiscard]] std::size_t filledBlocks() const noexcept { return head.load(std::memory_order_acquire) - tail.load(std::memory_order_acquire); }
    [[nodiscard]] float*      slotSamples(std::uint32_t index) noexcept { return samples.data() + static_cast<std::size_t>(index % blockCount) * channelStride * kBlockFrames; }

    template<std::invocable<const float*, std::size_t, std::size_t> TConsume>
    void drain(TConsume&& consume) {
        for (std::uint32_t index = tail.load(std::memory_order_relaxed); index != head.load(std::memory_order_acquire); ++index) {
            consume(slotSamples(index), static_cast<std::size_t>(blockFrames[index % blockCount]), static_cast<std::size_t>(blockChannels[index % blockCount]));
            tail.store(index + 1U, std::memory_order_release);
        }
    }

    template<std::predicate<float*, std::size_t, std::size_t> TProduce>
    void fill(std::size_t targetBlocks, TProduce&& produce) {
        for (std::uint32_t index = head.load(std::memory_order_relaxed); index - tail.load(std::memory_order_acquire) < std::min<std::size_t>(targetBlocks, blockCount); ++index) {
            if (!produce(slotSamples(index), kBlockFrames, static_cast<std::size_t>(channelStride))) {
                return;
            }
            blockFrames[index % blockCount]   = static_cast<std::uint32_t>(kBlockFrames);
            blockChannels[index % blockCount] = channelStride;
            head.store(index + 1U, std::memory_order_release);
        }
    }
};

template<AudioSample T>
struct AudioSinkState : AudioStateBase<T> {
    std::atomic<bool> intentionalSilence{true};

    template<typename InputSpan>
    [[nodiscard]] std::size_t writeFromInput(const InputSpan& inSpan, std::size_t channelCount) {
        const std::size_t nSamples = std::min(wholeFrameSamples(inSpan.size(), channelCount), wholeFrameSamples(this->writer.available(), channelCount));
        if (this->stopRequested.load(std::memory_order_acquire) || nSamples == 0UZ) {
            return 0UZ;
        }
        auto span = this->writer.tryReserve(nSamples);
        if (span.empty()) {
            return 0UZ;
        }
        std::copy_n(inSpan.begin(), static_cast<std::ptrdiff_t>(nSamples), span.begin());
        span.publish(nSamples);
        intentionalSilence.store(false, std::memory_order_release);
        return nSamples;
    }

    void readPlanarFloat(float* output, std::size_t frameCount, std::size_t channelCount, std::size_t outputChannels = 0UZ) {
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
        outputChannels                        = outputChannels == 0UZ ? alignedChannelCount : outputChannels;
        const std::size_t available           = this->reader.available();
        const std::size_t alignedAvailable    = wholeFrameSamples(available, alignedChannelCount);
        const std::size_t sampleCount         = frameCount * alignedChannelCount;
        const std::size_t nRead               = std::min(sampleCount, alignedAvailable);
        const std::size_t readFrames          = nRead / alignedChannelCount;

        if (nRead > 0U) {
            auto       read    = this->reader.get(nRead);
            const auto padding = this->channelPadding.load(std::memory_order_acquire);
            for (std::size_t channel = 0UZ; channel < outputChannels; ++channel) {
                const std::optional<std::size_t> source = paddingSourceChannel(channel, alignedChannelCount, padding);
                for (std::size_t frame = 0UZ; frame < readFrames; ++frame) {
                    output[channel * frameCount + frame] = source ? toFloatSample(read[frame * alignedChannelCount + *source]) : 0.f;
                }
            }
            std::ignore = read.consume(nRead);
        }

        if (readFrames < frameCount && !intentionalSilence.load(std::memory_order_relaxed)) {
            this->underrunCount.fetch_add(1U, std::memory_order_relaxed);
        }
        for (std::size_t channel = 0U; channel < outputChannels; ++channel) {
            std::fill_n(output + static_cast<std::ptrdiff_t>(channel * frameCount + readFrames), static_cast<std::ptrdiff_t>(frameCount - readFrames), 0.0f);
        }
    }

    void feedWorklet(WorkletBlockRing& ring, std::size_t targetBlocks) {
        const std::size_t missed = ring.missedBlocks.exchange(0U, std::memory_order_acq_rel);
        if (!intentionalSilence.load(std::memory_order_acquire)) {
            this->underrunCount.fetch_add(missed, std::memory_order_relaxed);
        }
        const std::size_t logicalChannels = this->numChannels.load(std::memory_order_acquire);
        ring.fill(targetBlocks, [&](float* block, std::size_t frames, std::size_t outputChannels) {
            const std::size_t staged     = this->reader.available();
            const bool        wholeBlock = staged >= frames * logicalChannels;
            if (logicalChannels == 0UZ || staged == 0UZ || (!wholeBlock && !intentionalSilence.load(std::memory_order_acquire))) {
                return false;
            }
            readPlanarFloat(block, frames, logicalChannels, outputChannels);
            return true;
        });
    }
};

template<AudioSample T>
struct AudioSourceState : AudioStateBase<T> {
    template<typename TRead>
    [[nodiscard]] std::size_t writeChannels(std::size_t frameCount, std::size_t inputChannels, std::size_t outputChannels, TRead&& readSample) {
        if (this->stopRequested.load(std::memory_order_acquire) || frameCount == 0UZ || inputChannels == 0UZ || outputChannels == 0UZ) {
            return 0UZ;
        }
        this->observedChannels.store(inputChannels, std::memory_order_relaxed);
        const std::size_t nFrames  = std::min(frameCount, this->writer.available() / outputChannels);
        std::size_t       nSamples = 0UZ;
        if (nFrames > 0UZ) {
            auto span = this->writer.tryReserve(nFrames * outputChannels);
            nSamples  = wholeFrameSamples(span.size(), outputChannels);
            adaptChannels<T>(std::span<T>(span.data(), nSamples), nSamples / outputChannels, inputChannels, outputChannels, this->channelPadding.load(std::memory_order_acquire), std::forward<TRead>(readSample));
            span.publish(nSamples);
        }
        if (nSamples < frameCount * outputChannels) {
            this->overflowCount.fetch_add(1UZ, std::memory_order_relaxed);
        }
        return nSamples;
    }

    [[nodiscard]] std::size_t writePlanarFloat(const float* input, std::size_t frameCount, std::size_t inputChannels, std::size_t outputChannels) {
        if (this->stopRequested.load(std::memory_order_acquire) || input == nullptr || frameCount == 0U) {
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
                return static_cast<std::int16_t>(std::lround(static_cast<double>(clamped) * static_cast<double>(std::numeric_limits<std::int16_t>::max())));
            }
        };

        return writeChannels(frameCount, inputChannels, outputChannels, [&](std::size_t frame, std::size_t channel) { return fromFloatSample(input[channel * frameCount + frame]); });
    }

    void drainWorklet(WorkletBlockRing& ring) {
        this->overflowCount.fetch_add(ring.missedBlocks.exchange(0U, std::memory_order_acq_rel), std::memory_order_relaxed);
        ring.drain([this](const float* block, std::size_t frames, std::size_t inputChannels) { std::ignore = writePlanarFloat(block, frames, inputChannels, this->numChannels.load(std::memory_order_acquire)); });
    }

    [[nodiscard]] std::size_t readToOutput(std::span<T> output, std::size_t channelCount) {
        const std::size_t alignedOutputSize = wholeFrameSamples(output.size(), channelCount);
        const std::size_t alignedAvailable  = wholeFrameSamples(this->reader.available(), channelCount);
        const std::size_t nSamplesToRead    = std::min(alignedOutputSize, alignedAvailable);
        if (nSamplesToRead == 0U) {
            return 0U;
        }

        auto readSpan = this->reader.get(nSamplesToRead);
        std::copy_n(readSpan.begin(), static_cast<std::ptrdiff_t>(readSpan.size()), output.begin());
        std::ignore = readSpan.consume(readSpan.size());
        return readSpan.size();
    }
};

} // namespace detail
} // namespace gr::audio

#endif // GNURADIO_AUDIO_BACKENDS_HPP
