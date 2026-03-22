#ifndef GNURADIO_AUDIO_FILE_SOURCE_HPP
#define GNURADIO_AUDIO_FILE_SOURCE_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>
#include <gnuradio-4.0/audio/AudioCommon.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <algorithm>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gr::audio {

namespace fileio = gr::algorithm::fileio;

namespace detail {

template<AudioSample T>
struct DecodedAudio {
    std::vector<T> pcm;
    gr::Size_t     channels   = 0U;
    float          sampleRate = 0.f;
};

template<AudioSample T>
inline std::expected<DecodedAudio<T>, gr::Error> decodeAudioBuffer(std::span<const std::uint8_t> bytes) {
    if (bytes.empty()) {
        return std::unexpected(gr::Error("audio input is empty"));
    }

    ma_decoder_config decoderConfig = ma_decoder_config_init(maFormatFor<T>(), 0, 0);
    ma_decoder        decoder{};
    ma_result         initResult = ma_decoder_init_memory(bytes.data(), bytes.size_bytes(), &decoderConfig, &decoder);
    if (initResult != MA_SUCCESS) {
        return std::unexpected(makeMiniaudioError("ma_decoder_init_memory()", initResult));
    }
    gr::on_scope_exit decoderGuard([&decoder] { ma_decoder_uninit(&decoder); });

    ma_format outputFormat = ma_format_unknown;
    ma_uint32 channels     = 0;
    ma_uint32 sampleRate   = 0;
    ma_result formatResult = ma_data_source_get_data_format(&decoder, &outputFormat, &channels, &sampleRate, nullptr, 0);
    if (formatResult != MA_SUCCESS) {
        return std::unexpected(makeMiniaudioError("ma_data_source_get_data_format()", formatResult));
    }
    if (channels == 0U || sampleRate == 0U) {
        return std::unexpected(gr::Error("decoded audio format is incomplete"));
    }
    if (outputFormat != maFormatFor<T>()) {
        return std::unexpected(gr::Error("decoded audio format does not match requested PCM type"));
    }

    constexpr std::size_t kChunkFrames = 1024U;
    std::vector<T>        chunk(kChunkFrames * channels);
    DecodedAudio<T>       decoded;
    decoded.channels   = static_cast<gr::Size_t>(channels);
    decoded.sampleRate = static_cast<float>(sampleRate);

    while (true) {
        ma_uint64 framesRead = 0;
        ma_result readResult = ma_decoder_read_pcm_frames(&decoder, chunk.data(), kChunkFrames, &framesRead);
        if (readResult != MA_SUCCESS && readResult != MA_AT_END) {
            return std::unexpected(makeMiniaudioError("ma_decoder_read_pcm_frames()", readResult));
        }

        const std::size_t sampleCount = static_cast<std::size_t>(framesRead) * static_cast<std::size_t>(channels);
        decoded.pcm.insert(decoded.pcm.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(sampleCount));

        if (readResult == MA_AT_END || framesRead == 0U) {
            break;
        }
    }

    return decoded;
}

} // namespace detail

GR_REGISTER_BLOCK(gr::audio::AudioFileSource, [T], [ float, int16_t ])

template<detail::AudioSample T>
struct AudioFileSource : gr::Block<AudioFileSource<T>> {
    using Description = Doc<R""(Loads and decodes audio files into memory using miniaudio library.
Supported formats: wav, flac, mp3.
Works with local files and HTTP/HTTPS URLs and supports repeat playback.

Loads the entire file into memory before playback starts, so it is not suitable for large files.
For streaming large WAV files without memory buffering, use `WavSource` instead.)"">;

    gr::PortOut<T> out;

    gr::Annotated<std::string, "uri", gr::Visible, gr::Doc<"Local path or HTTP/HTTPS URL">>          uri;
    gr::Annotated<bool, "repeat", gr::Doc<"Repeat decoded PCM when end-of-file is reached">>         repeat      = false;
    gr::Annotated<float, "sample_rate", gr::Visible, gr::Unit<"Hz">, gr::Doc<"Decoded sample rate">> sample_rate = 0.f;
    gr::Annotated<gr::Size_t, "channels", gr::Visible, gr::Doc<"Decoded interleaved channel count">> channels    = 0U;

    GR_MAKE_REFLECTABLE(AudioFileSource, out, uri, repeat, sample_rate, channels);

    fileio::Reader            _reader;
    std::vector<std::uint8_t> _accumulatedBytes;
    std::vector<T>            _pcm;
    std::size_t               _sampleCursor{0U};
    bool                      _formatTagPending{true};
    bool                      _readerFinalSeen{false};
    bool                      _decoded{false};
    std::optional<gr::Error>  _readerError;

    using gr::Block<AudioFileSource<T>>::Block;

    void start() {
        auto readerExp = fileio::readAsync(uri.value, fileio::ReaderConfig{});
        if (!readerExp.has_value()) {
            throw gr::exception(readerExp.error().message, readerExp.error().sourceLocation);
        }
        _reader = std::move(readerExp.value());
        _accumulatedBytes.clear();
        _pcm.clear();
        _sampleCursor     = 0U;
        _formatTagPending = true;
        _readerFinalSeen  = false;
        _decoded          = false;
        _readerError.reset();
        sample_rate = 0.f;
        channels    = 0U;
    }

    void stop() {
        _reader.cancel();
        _reader = {};
        _accumulatedBytes.clear();
        _pcm.clear();
        _sampleCursor     = 0U;
        _formatTagPending = true;
        _readerFinalSeen  = false;
        _decoded          = false;
        _readerError.reset();
    }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (newSettings.contains("uri") && oldSettings.at("uri") != newSettings.at("uri")) {
            stop();
            start();
        }
    }

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        if (outSpan.empty()) {
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        if (!_readerFinalSeen) {
            _reader.poll([this](const auto& res) {
                if (res.isFinal) {
                    _readerFinalSeen = true;
                }
                if (res.data.has_value()) {
                    const auto chunk = res.data.value();
                    if (!chunk.empty()) {
                        _accumulatedBytes.insert(_accumulatedBytes.end(), chunk.begin(), chunk.end());
                    }
                } else if (!_readerError.has_value()) {
                    _readerError = res.data.error();
                }
            });
        }

        if (_readerFinalSeen) {
            if (_readerError.has_value()) {
                throw gr::exception(_readerError->message, _readerError->sourceLocation);
            }

            if (!_decoded) {
                if (_accumulatedBytes.empty()) {
                    throw gr::exception("audio file is empty");
                }

                auto decodedExp = detail::decodeAudioBuffer<T>(_accumulatedBytes);
                if (!decodedExp.has_value()) {
                    throw gr::exception(decodedExp.error().message, decodedExp.error().sourceLocation);
                }

                _pcm        = std::move(decodedExp->pcm);
                sample_rate = decodedExp->sampleRate;
                channels    = decodedExp->channels;
                _decoded    = true;
                _accumulatedBytes.clear();
            }
        }

        if (!_decoded) {
            outSpan.publish(0U);
            return gr::work::Status::OK;
        }

        if (_pcm.empty()) {
            outSpan.publish(0U);
            return gr::work::Status::DONE;
        }

        if (_sampleCursor >= _pcm.size()) {
            if (!repeat.value) {
                outSpan.publish(0U);
                return gr::work::Status::DONE;
            }
            _sampleCursor = 0U;
        }

        const std::size_t remaining  = _pcm.size() - _sampleCursor;
        const std::size_t nPublished = std::min(outSpan.size(), remaining);
        std::copy_n(_pcm.begin() + static_cast<std::ptrdiff_t>(_sampleCursor), static_cast<std::ptrdiff_t>(nPublished), outSpan.begin());
        _sampleCursor += nPublished;

        if (nPublished > 0U && _formatTagPending) {
            auto tagMap = property_map{};
            gr::tag::put(tagMap, gr::tag::SAMPLE_RATE, sample_rate.value);
            gr::tag::put(tagMap, tag::AUDIO_CHANNELS, channels.value);
            gr::tag::put(tagMap, gr::tag::SIGNAL_NAME, uri.value);
            outSpan.publishTag(std::move(tagMap), 0U);
            _formatTagPending = false;
        }

        outSpan.publish(nPublished);
        if (nPublished == 0U) {
            return gr::work::Status::DONE;
        }
        if (!repeat.value && _sampleCursor >= _pcm.size()) {
            return gr::work::Status::DONE;
        }
        return gr::work::Status::OK;
    }
};

static_assert(gr::BlockLike<AudioFileSource<float>>);

} // namespace gr::audio

#endif // GNURADIO_AUDIO_FILE_SOURCE_HPP
