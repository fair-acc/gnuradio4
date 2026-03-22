#ifndef GNURADIO_AUDIO_BLOCKS_HPP
#define GNURADIO_AUDIO_BLOCKS_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/audio/AudioCommon.hpp>
#include <gnuradio-4.0/audio/AudioFileSource.hpp>
#include <gnuradio-4.0/audio/WavSource.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
#include <miniaudio.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#if defined(__EMSCRIPTEN__)
#include <emscripten/threading.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace gr::audio {

enum class Backend {
    auto_detect,
    null_backend,
};

GR_REGISTER_BLOCK(gr::audio::AudioSink, [T], [ float, int16_t ])

template<detail::AudioSample T>
struct AudioSink : gr::Block<AudioSink<T>> {
    using Description = Doc<R""(Play interleaved PCM samples on the system audio output device using miniaudio.

`AudioSink` is intentionally PCM-only. It does not parse container formats or fetch files.
Pair it with `WavSource`, `AudioFileSource`, or any other block that already produces decoded PCM.)"">;

    gr::PortIn<T> in;

    gr::Annotated<float, "sample_rate", gr::Visible, gr::Unit<"Hz">, gr::Doc<"PCM sample rate. Updated automatically; not intended to be set by the user.">>        sample_rate    = 48000.f;
    gr::Annotated<gr::Size_t, "audio_channels", gr::Visible, gr::Doc<"PCM interleaved channel count. Updated automatically; not intended to be set by the user..">> audio_channels = 1U;
    gr::Annotated<gr::Size_t, "buffer_frames", gr::Visible, gr::Doc<"Software queue depth in PCM frames">>                                                          buffer_frames  = 4096U;
    gr::Annotated<Backend, "backend", gr::Visible, gr::Doc<"Playback backend: auto_detect or null_backend">>                                                        backend        = Backend::auto_detect;

    GR_MAKE_REFLECTABLE(AudioSink, in, sample_rate, audio_channels, buffer_frames, backend);

    using gr::Block<AudioSink<T>>::Block;
    using SampleBuffer = gr::CircularBuffer<T, std::dynamic_extent, gr::ProducerType::Single>;
    using SampleWriter = decltype(std::declval<SampleBuffer&>().new_writer());
    using SampleReader = decltype(std::declval<SampleBuffer&>().new_reader());

    ma_context             _context{};
    ma_device              _device{};
    bool                   _contextInitialised{false};
    bool                   _deviceInitialised{false};
    std::atomic<bool>      _stopRequested{false};
    std::atomic<ma_uint32> _activeChannelCount{1U};

    std::mutex   _deviceMutex; // Used for start/stop/reconfigure.
    SampleBuffer _buffer{1U};
    SampleWriter _writer{_buffer.new_writer()};
    SampleReader _reader{_buffer.new_reader()};

    void start() {
        std::lock_guard deviceLock(_deviceMutex);
        initialiseDeviceUnlocked();
    }

    void stop() { shutdownDevice(); }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        const bool sampleRateChanged   = newSettings.contains("sample_rate") && oldSettings.at("sample_rate") != newSettings.at("sample_rate");
        const bool channelCountChanged = newSettings.contains("audio_channels") && oldSettings.at("audio_channels") != newSettings.at("audio_channels");
        const bool bufferFramesChanged = newSettings.contains("buffer_frames") && oldSettings.at("buffer_frames") != newSettings.at("buffer_frames");
        const bool backendChanged      = newSettings.contains("backend") && oldSettings.at("backend") != newSettings.at("backend");

        if (sampleRateChanged || channelCountChanged || bufferFramesChanged || backendChanged) {
            std::lock_guard deviceLock(_deviceMutex);
            initialiseDeviceUnlocked();
        }
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan) {
        if (inSpan.empty()) {
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        const std::size_t channelCount  = std::max<std::size_t>(1U, static_cast<std::size_t>(audio_channels.value));
        const std::size_t nFrameSamples = inSpan.size() - (inSpan.size() % channelCount);
        if (nFrameSamples == 0U) {
            std::ignore = inSpan.consume(0U);
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        std::size_t nWritten = 0U;
        if (!_stopRequested.load(std::memory_order_acquire)) {
            const std::size_t available = _writer.available();
            if (available > 0U) {
                const std::size_t   nChunk     = std::min(nFrameSamples, available);
                WriterSpanLike auto writerSpan = _writer.tryReserve(nChunk);
                if (!writerSpan.empty()) {
                    std::copy_n(inSpan.begin(), static_cast<std::ptrdiff_t>(writerSpan.size()), writerSpan.begin());
                    writerSpan.publish(writerSpan.size());
                    nWritten = writerSpan.size();
                }
            }
        }

        std::ignore = inSpan.consume(nWritten);
        return gr::work::Status::OK;
    }

private:
#if defined(__EMSCRIPTEN__)
    struct MainThreadTask {
        AudioSink*         self{nullptr};
        std::exception_ptr exception{};
    };

    static void initialiseDeviceOnMainThread(void* opaque) {
        auto* task = static_cast<MainThreadTask*>(opaque);
        if (task == nullptr || task->self == nullptr) {
            return;
        }
        try {
            task->self->initialiseDeviceUnlockedImpl();
        } catch (...) {
            task->exception = std::current_exception();
        }
    }

    static void shutdownDeviceOnMainThread(void* opaque) {
        auto* task = static_cast<MainThreadTask*>(opaque);
        if (task == nullptr || task->self == nullptr) {
            return;
        }
        try {
            task->self->shutdownDeviceUnlockedImpl();
        } catch (...) {
            task->exception = std::current_exception();
        }
    }

    void initialiseDeviceUnlocked() {
        MainThreadTask task{.self = this};
        if (emscripten_is_main_runtime_thread()) {
            initialiseDeviceOnMainThread(&task);
        } else {
            emscripten_sync_run_in_main_runtime_thread(EM_FUNC_SIG_VI, initialiseDeviceOnMainThread, &task);
        }
        if (task.exception) {
            std::rethrow_exception(task.exception);
        }
    }

    void shutdownDeviceUnlocked() {
        MainThreadTask task{.self = this};
        if (emscripten_is_main_runtime_thread()) {
            shutdownDeviceOnMainThread(&task);
        } else {
            emscripten_sync_run_in_main_runtime_thread(EM_FUNC_SIG_VI, shutdownDeviceOnMainThread, &task);
        }
        if (task.exception) {
            std::rethrow_exception(task.exception);
        }
    }
#else
    void initialiseDeviceUnlocked() { initialiseDeviceUnlockedImpl(); }
    void shutdownDeviceUnlocked() { shutdownDeviceUnlockedImpl(); }
#endif

    [[nodiscard]] ma_uint32 currentSampleRate() const {
        const auto rounded = std::lround(static_cast<double>(sample_rate.value));
        return static_cast<ma_uint32>(std::max<long>(1L, rounded));
    }

    [[nodiscard]] ma_uint32 currentChannelCount() const { return static_cast<ma_uint32>(std::max<gr::Size_t>(1U, audio_channels.value)); }

    [[nodiscard]] std::size_t bufferCapacitySamples(ma_uint32 channelCount) const { return static_cast<std::size_t>(std::max<gr::Size_t>(1U, buffer_frames.value)) * static_cast<std::size_t>(std::max<ma_uint32>(1U, channelCount)); }

    void recreateBuffer(std::size_t capacitySamples) {
        _buffer = SampleBuffer(std::max<std::size_t>(1U, capacitySamples));
        _writer = _buffer.new_writer();
        _reader = _buffer.new_reader();
    }

    void initialiseDeviceUnlockedImpl() {
        shutdownDeviceUnlockedImpl();

        const ma_uint32 sampleRateValue = currentSampleRate();
        const ma_uint32 channelCount    = currentChannelCount();
        if (sampleRateValue == 0U || channelCount == 0U) {
            throw gr::exception("AudioSink requires sample_rate > 0 and audio_channels > 0");
        }

        ma_context_config contextConfig = ma_context_config_init();
        ma_result         contextResult = MA_ERROR;
        if (backend.value == Backend::null_backend) {
            const ma_backend forcedBackend = ma_backend_null;
            contextResult                  = ma_context_init(&forcedBackend, 1, &contextConfig, &_context);
        } else {
            contextResult = ma_context_init(nullptr, 0, &contextConfig, &_context);
        }
        if (contextResult != MA_SUCCESS) {
            throw gr::exception(detail::makeMiniaudioError("ma_context_init()", contextResult).message);
        }
        _contextInitialised = true;

        recreateBuffer(bufferCapacitySamples(channelCount));
        _stopRequested.store(false, std::memory_order_release);

        ma_device_config deviceConfig  = ma_device_config_init(ma_device_type_playback);
        deviceConfig.playback.format   = detail::maFormatFor<T>();
        deviceConfig.playback.channels = channelCount;
        deviceConfig.sampleRate        = sampleRateValue;
        deviceConfig.dataCallback      = &AudioSink::audioCallback;
        deviceConfig.pUserData         = this;

        const ma_result deviceResult = ma_device_init(&_context, &deviceConfig, &_device);
        if (deviceResult != MA_SUCCESS) {
            shutdownDeviceUnlockedImpl();
            throw gr::exception(detail::makeMiniaudioError("ma_device_init()", deviceResult).message);
        }
        _deviceInitialised = true;
        _activeChannelCount.store(channelCount, std::memory_order_relaxed);

        const ma_result startResult = ma_device_start(&_device);
        if (startResult != MA_SUCCESS) {
            shutdownDeviceUnlockedImpl();
            throw gr::exception(detail::makeMiniaudioError("ma_device_start()", startResult).message);
        }
    }

    void shutdownDevice() {
        _stopRequested.store(true, std::memory_order_release);
        std::lock_guard deviceLock(_deviceMutex);
        shutdownDeviceUnlocked();
    }

    void shutdownDeviceUnlockedImpl() {
        _stopRequested.store(true, std::memory_order_release);

        if (_deviceInitialised) {
            ma_device_uninit(&_device);
            _deviceInitialised = false;
        }
        _activeChannelCount.store(1U, std::memory_order_relaxed);
        if (_contextInitialised) {
            ma_context_uninit(&_context);
            _contextInitialised = false;
        }
        recreateBuffer(1U);
    }

    static void audioCallback(ma_device* device, void* output, const void* /*input*/, ma_uint32 frameCount) {
        auto* self = static_cast<AudioSink*>(device->pUserData);
        if (self == nullptr) {
            return;
        }

        const std::size_t sampleCount = static_cast<std::size_t>(frameCount) * static_cast<std::size_t>(self->_activeChannelCount.load(std::memory_order_relaxed));
        auto              outputSpan  = std::span<T>(static_cast<T*>(output), sampleCount);
        const std::size_t nRead       = std::min(outputSpan.size(), self->_reader.available());
        if (nRead > 0U) {
            auto read = self->_reader.get(nRead);
            std::copy_n(read.begin(), static_cast<std::ptrdiff_t>(nRead), outputSpan.begin());
            std::ignore = read.consume(nRead);
        }
        std::fill(outputSpan.begin() + static_cast<std::ptrdiff_t>(nRead), outputSpan.end(), T{});
    }
};

static_assert(gr::BlockLike<AudioSink<float>>);

} // namespace gr::audio

#endif // GNURADIO_AUDIO_BLOCKS_HPP
