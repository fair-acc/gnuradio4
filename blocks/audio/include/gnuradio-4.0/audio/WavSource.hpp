#ifndef GNURADIO_AUDIO_WAV_SOURCE_HPP
#define GNURADIO_AUDIO_WAV_SOURCE_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>
#include <gnuradio-4.0/audio/AudioBackends.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace gr::audio {
namespace detail {

[[nodiscard]] inline std::uint16_t readLe16(std::span<const std::uint8_t> bytes) {
    const auto value = static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8U);
    return static_cast<std::uint16_t>(value);
}

[[nodiscard]] inline std::uint32_t readLe32(std::span<const std::uint8_t> bytes) { return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8U) | (static_cast<std::uint32_t>(bytes[2]) << 16U) | (static_cast<std::uint32_t>(bytes[3]) << 24U); }

[[nodiscard]] inline bool bytesEqual(std::span<const std::uint8_t> bytes, std::string_view text) { return bytes.size() == text.size() && std::equal(bytes.begin(), bytes.end(), text.begin(), text.end()); }

struct WavFormat {
    std::uint16_t formatTag     = 0U;
    std::uint16_t channels      = 0U;
    std::uint32_t sampleRate    = 0U;
    std::uint16_t blockAlign    = 0U;
    std::uint16_t bitsPerSample = 0U;

    [[nodiscard]] std::size_t bytesPerSample() const { return static_cast<std::size_t>(bitsPerSample) / 8U; }
};

inline std::expected<WavFormat, gr::Error> parseWavFmtChunk(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < 16U) {
        return std::unexpected(gr::Error("WAV fmt chunk is too small"));
    }

    WavFormat  format;
    const auto rawFormatTag = readLe16(bytes.subspan(0U, 2U));
    format.channels         = readLe16(bytes.subspan(2U, 2U));
    format.sampleRate       = readLe32(bytes.subspan(4U, 4U));
    format.blockAlign       = readLe16(bytes.subspan(12U, 2U));
    format.bitsPerSample    = readLe16(bytes.subspan(14U, 2U));

    if (format.channels == 0U || format.sampleRate == 0U || format.bitsPerSample == 0U) {
        return std::unexpected(gr::Error("WAV fmt chunk is incomplete"));
    }
    if ((format.bitsPerSample % 8U) != 0U) {
        return std::unexpected(gr::Error("WAV bits_per_sample must be byte aligned"));
    }

    format.formatTag = rawFormatTag;
    if (rawFormatTag == 0xFFFEU) {
        if (bytes.size() < 40U) {
            return std::unexpected(gr::Error("WAV extensible fmt chunk is too small"));
        }
        format.formatTag = readLe16(bytes.subspan(24U, 2U));
    }

    const std::size_t expectedBlockAlign = static_cast<std::size_t>(format.channels) * format.bytesPerSample();
    if (format.blockAlign == 0U || static_cast<std::size_t>(format.blockAlign) != expectedBlockAlign) {
        return std::unexpected(gr::Error("WAV block_align does not match channels and bits_per_sample"));
    }

    const bool supportedPcm   = format.formatTag == 1U && (format.bitsPerSample == 8U || format.bitsPerSample == 16U || format.bitsPerSample == 24U || format.bitsPerSample == 32U);
    const bool supportedFloat = format.formatTag == 3U && format.bitsPerSample == 32U;
    if (!supportedPcm && !supportedFloat) {
        return std::unexpected(gr::Error(std::format("unsupported WAV format tag {} with {} bits per sample", format.formatTag, format.bitsPerSample)));
    }

    return format;
}

template<AudioSample T>
[[nodiscard]] T convertNormalisedSample(double value) {
    const double clamped = std::clamp(value, -1.0, 1.0);
    if constexpr (std::same_as<T, float>) {
        return static_cast<float>(clamped);
    } else {
        if (clamped <= -1.0) {
            return std::numeric_limits<std::int16_t>::min();
        }
        const auto scaled = std::lround(clamped * static_cast<double>(std::numeric_limits<std::int16_t>::max()));
        return static_cast<std::int16_t>(std::clamp<long>(static_cast<long>(scaled), std::numeric_limits<std::int16_t>::min(), std::numeric_limits<std::int16_t>::max()));
    }
}

template<AudioSample T>
[[nodiscard]] T decodeWavSample(std::span<const std::uint8_t> bytes, const WavFormat& format) {
    if constexpr (std::same_as<T, std::int16_t>) {
        if (format.formatTag == 1U && format.bitsPerSample == 16U) {
            return static_cast<std::int16_t>(readLe16(bytes.first(2U)));
        }
    }

    if (format.formatTag == 3U) {
        const float value = std::bit_cast<float>(readLe32(bytes.first(4U)));
        return convertNormalisedSample<T>(static_cast<double>(value));
    }

    double normalised = 0.0;
    if (format.bitsPerSample == 8U) {
        normalised = static_cast<double>(static_cast<int>(bytes[0]) - 128) / 128.0;
    } else if (format.bitsPerSample == 16U) {
        const auto signedValue = static_cast<std::int16_t>(readLe16(bytes.first(2U)));
        normalised             = static_cast<double>(signedValue) / 32768.0;
    } else if (format.bitsPerSample == 24U) {
        const auto   assembled   = static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8U) | (static_cast<std::uint32_t>(bytes[2]) << 16U);
        std::int32_t signedValue = static_cast<std::int32_t>(assembled);
        if ((bytes[2] & 0x80U) != 0U) {
            signedValue |= static_cast<std::int32_t>(0xFF00'0000U);
        }
        normalised = static_cast<double>(signedValue) / 8388608.0;
    } else if (format.bitsPerSample == 32U) {
        const auto signedValue = static_cast<std::int32_t>(readLe32(bytes.first(4U)));
        normalised             = static_cast<double>(signedValue) / 2147483648.0;
    }

    return convertNormalisedSample<T>(normalised);
}

} // namespace detail

GR_REGISTER_BLOCK(gr::audio::WavSource, [T], [ float, int16_t ])

template<detail::AudioSample T>
struct WavSource : gr::Block<WavSource<T>> {
    using Description = Doc<R""(
Streams RIFF/WAVE audio from local paths or HTTP/HTTPS URLs without seeking.

Supported WAV encodings:
- PCM: 8-bit, 16-bit, 24-bit, 32-bit
- IEEE float: 32-bit
- Interleaved mono or multi-channel audio

Limitations:
- Compressed WAV formats are not supported (for example ADPCM, mu-law, A-law, or MP3-in-WAV)
- Forward-only streaming only: no seeking, rewind, or repeat
- Requires a fmt chunk before the data chunk
- Stops at the first data chunk and ignores trailing RIFF chunks

Outputs interleaved samples as either normalized float values or signed 16-bit PCM,
depending on the block type. Publishes sample rate, channel count, and signal name as tags.)"">;

    gr::PortOut<T> out;

    gr::Annotated<std::string, "uri", gr::Visible, gr::Doc<"Local path or HTTP/HTTPS URL to a WAV file">>             uri;
    gr::Annotated<float, "sample_rate", gr::Visible, gr::Unit<"Hz">, gr::Doc<"Decoded sample rate, updated on load">> sample_rate = 0.f;
    gr::Annotated<gr::Size_t, "channels", gr::Visible, gr::Doc<"Decoded interleaved channel count, updated on load">> channels    = 0U;

    GR_MAKE_REFLECTABLE(WavSource, out, uri, sample_rate, channels);

    static constexpr std::size_t kMaxHeaderBytes = 1024U * 1024U;

    struct DecodeResult {
        std::size_t samplesWritten{0U};
        std::size_t bytesConsumed{0U};
    };

    gr::algorithm::fileio::Reader _reader;
    bool                          _headerParsed{false};
    bool                          _readerFinalSeen{false};
    bool                          _failed{false};
    detail::WavFormat             _format{};
    std::size_t                   _dataBytesRemaining{0U};
    std::vector<std::uint8_t>     _headerBuffer;
    std::array<std::uint8_t, 4>   _partialSample{};
    std::size_t                   _partialSampleSize{0U};
    bool                          _formatTagPending{true};

    using gr::Block<WavSource<T>>::Block;

    void resetState() {
        _reader          = {};
        _headerParsed    = false;
        _readerFinalSeen = false;
        _failed          = false;
        _format          = {};
        sample_rate      = 0.f;
        channels         = 0U;
        _headerBuffer.clear();
        _dataBytesRemaining = 0U;
        _partialSampleSize  = 0U;
        _formatTagPending   = true;
    }

    void fail(std::string_view endpoint, gr::Error error) {
        this->emitErrorMessage(endpoint, error);
        _reader.cancel();
        resetState();
        _readerFinalSeen = true;
        _failed          = true;
    }

    void start() {
        resetState();

        gr::algorithm::fileio::ReaderConfig config{};
        config.chunkBytes          = std::max<std::size_t>(1U, out.max_buffer_size() * sizeof(T) / 4U);
        config.chunkAlignmentBytes = 1U;

        auto readerExp = gr::algorithm::fileio::readAsync(uri.value, std::move(config));
        if (!readerExp) {
            fail("WavSource::start()", readerExp.error());
            return;
        }
        _reader = std::move(readerExp.value());
    }

    void stop() {
        _reader.cancel();
        resetState();
    }

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        if (outSpan.empty()) {
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        if (_failed) {
            outSpan.publish(0U);
            return gr::work::Status::ERROR;
        }

        auto                       output    = std::span<T>(outSpan);
        std::size_t                published = 0U;
        std::optional<gr::Error>   error;
        std::optional<std::size_t> requiredOutputSize;
        bool                       justParsedHeader = false;

        if (!_headerParsed) {
            _reader.poll(
                [&](const auto& res) {
                    if (res.isFinal) {
                        _readerFinalSeen = true;
                    }

                    if (res.requiredOutputSize) {
                        requiredOutputSize = res.requiredOutputSize;
                    }

                    if (!res.data) {
                        if (!res.requiredOutputSize) {
                            error = res.data.error();
                        }
                        return;
                    }

                    const auto chunk = res.data.value();
                    if (!chunk.empty()) {
                        _headerBuffer.insert(_headerBuffer.end(), chunk.begin(), chunk.end());
                        if (_headerBuffer.size() > kMaxHeaderBytes) {
                            error = gr::Error("WAV header exceeds 1MB");
                        }
                    }
                },
                std::numeric_limits<std::size_t>::max(), false);

            if (error) {
                fail("WavSource::processBulk()", *error);
                outSpan.publish(0U);
                return gr::work::Status::ERROR;
            }

            auto headerParsedExp = tryParseHeader();
            if (!headerParsedExp) {
                fail("WavSource::processBulk()", headerParsedExp.error());
                outSpan.publish(0U);
                return gr::work::Status::ERROR;
            }
            _headerParsed    = *headerParsedExp;
            justParsedHeader = _headerParsed;

            if (!_headerParsed && _readerFinalSeen) {
                fail("WavSource::processBulk()", gr::Error("WAV stream ended before data chunk"));
                outSpan.publish(0U);
                return gr::work::Status::ERROR;
            }
        }

        if (_headerParsed && !_headerBuffer.empty() && published < output.size()) {
            const std::size_t bytesToDecode = std::min(_headerBuffer.size(), _dataBytesRemaining);

            if (bytesToDecode > 0U) {
                const auto decoded = decodeChunk(std::span<const std::uint8_t>(_headerBuffer).first(bytesToDecode), output.subspan(published));
                published += decoded.samplesWritten;

                if (decoded.bytesConsumed > 0U) {
                    _headerBuffer.erase(_headerBuffer.begin(), _headerBuffer.begin() + static_cast<std::ptrdiff_t>(decoded.bytesConsumed));
                }
            }

            if (_dataBytesRemaining == 0U && !_headerBuffer.empty()) {
                _headerBuffer.clear();
            }
        }

        if (_headerParsed && !justParsedHeader && _headerBuffer.empty() && published < output.size() && _dataBytesRemaining > 0U) {
            const std::size_t maxBytes = maxBytesForOutputSamples(output.size() - published);

            _reader.poll(
                [&](const auto& res) {
                    if (res.isFinal) {
                        _readerFinalSeen = true;
                    }

                    if (res.requiredOutputSize) {
                        requiredOutputSize = res.requiredOutputSize;
                    }

                    if (!res.data) {
                        if (!res.requiredOutputSize) {
                            error = res.data.error();
                        }
                        return;
                    }

                    const auto chunk = res.data.value();
                    if (!chunk.empty()) {
                        const auto toConsume = std::min(chunk.size(), _dataBytesRemaining);
                        const auto decoded   = decodeChunk(chunk.first(toConsume), output.subspan(published));
                        published += decoded.samplesWritten;

                        if (decoded.bytesConsumed < toConsume) {
                            const auto unconsumed = chunk.subspan(decoded.bytesConsumed, toConsume - decoded.bytesConsumed);
                            _headerBuffer.assign(unconsumed.begin(), unconsumed.end());
                        }
                    }
                },
                maxBytes, false);

            if (error) {
                fail("WavSource::processBulk()", *error);
                outSpan.publish(0U);
                return gr::work::Status::ERROR;
            }
        }

        if (_readerFinalSeen) {
            if (_partialSampleSize > 0U) {
                fail("WavSource::processBulk()", gr::Error("WAV data ended mid-sample"));
                outSpan.publish(0U);
                return gr::work::Status::ERROR;
            }
            if (_dataBytesRemaining > 0U && _headerBuffer.empty()) {
                fail("WavSource::processBulk()", gr::Error("WAV data chunk truncated"));
                outSpan.publish(0U);
                return gr::work::Status::ERROR;
            }
        }

        if (published > 0U && _formatTagPending) {
            publishFormatTag(outSpan, 0U);
            _formatTagPending = false;
        }

        outSpan.publish(published);

        if (requiredOutputSize && published == 0U) {
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        if (_headerParsed && _dataBytesRemaining == 0U) {
            if (_partialSampleSize > 0U) {
                fail("WavSource::processBulk()", gr::Error("WAV data ended mid-sample"));
                outSpan.publish(0U);
                return gr::work::Status::ERROR;
            }
            return gr::work::Status::DONE;
        }

        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] std::expected<bool, gr::Error> tryParseHeader() {
        const auto bytes = std::span<const std::uint8_t>(_headerBuffer);

        if (bytes.size() < 12U) {
            return false;
        }

        if (!detail::bytesEqual(bytes.subspan(0U, 4U), "RIFF") || !detail::bytesEqual(bytes.subspan(8U, 4U), "WAVE")) {
            return std::unexpected(gr::Error("Not a valid WAVE file"));
        }

        std::size_t offset  = 12U;
        bool        haveFmt = false;

        while (offset + 8U <= bytes.size()) {
            const auto chunkId   = bytes.subspan(offset, 4U);
            const auto chunkSize = static_cast<std::size_t>(detail::readLe32(bytes.subspan(offset + 4U, 4U)));
            offset += 8U;

            if (detail::bytesEqual(chunkId, "data")) {
                if (!haveFmt) {
                    return std::unexpected(gr::Error("WAV data chunk before fmt chunk"));
                }
                if ((chunkSize % _format.blockAlign) != 0U) {
                    return std::unexpected(gr::Error("WAV data chunk size is not a multiple of block_align"));
                }
                _dataBytesRemaining = chunkSize;

                if (offset < bytes.size()) {
                    std::vector<std::uint8_t> crossover(bytes.begin() + static_cast<std::ptrdiff_t>(offset), bytes.end());
                    _headerBuffer = std::move(crossover);
                } else {
                    _headerBuffer.clear();
                }

                return true;
            }

            const std::size_t paddedSize = chunkSize + (chunkSize & 1U);
            if (offset + paddedSize > bytes.size()) {
                return false;
            }

            if (detail::bytesEqual(chunkId, "fmt ")) {
                auto fmtExp = detail::parseWavFmtChunk(bytes.subspan(offset, chunkSize));
                if (!fmtExp) {
                    return std::unexpected(fmtExp.error());
                }
                _format     = fmtExp.value();
                sample_rate = static_cast<float>(_format.sampleRate);
                channels    = static_cast<gr::Size_t>(_format.channels);
                haveFmt     = true;
            }

            offset += paddedSize;
        }

        return false;
    }

    [[nodiscard]] DecodeResult decodeChunk(std::span<const std::uint8_t> bytes, std::span<T> output) {
        DecodeResult result{};
        if (bytes.empty() || output.empty()) {
            return result;
        }

        const std::size_t bytesPerSample = _format.bytesPerSample();
        if (bytesPerSample == 0U) {
            return result;
        }

        std::size_t offset = 0U;

        if (_partialSampleSize > 0U) {
            const std::size_t needed    = bytesPerSample - _partialSampleSize;
            const std::size_t available = std::min(needed, bytes.size());

            std::copy_n(bytes.begin(), static_cast<std::ptrdiff_t>(available), _partialSample.begin() + static_cast<std::ptrdiff_t>(_partialSampleSize));
            _partialSampleSize += available;
            offset += available;
            result.bytesConsumed += available;
            _dataBytesRemaining -= available;

            if (_partialSampleSize == bytesPerSample) {
                output[result.samplesWritten++] = detail::decodeWavSample<T>(std::span(_partialSample).first(bytesPerSample), _format);
                _partialSampleSize              = 0U;
            } else {
                return result;
            }
        }

        const std::size_t maxSamples = std::min(output.size() - result.samplesWritten, (bytes.size() - offset) / bytesPerSample);
        for (std::size_t i = 0U; i < maxSamples; ++i) {
            output[result.samplesWritten++] = detail::decodeWavSample<T>(bytes.subspan(offset, bytesPerSample), _format);
            offset += bytesPerSample;
            result.bytesConsumed += bytesPerSample;
            _dataBytesRemaining -= bytesPerSample;
        }

        const std::size_t remaining = bytes.size() - offset;
        if (remaining > 0U && remaining < bytesPerSample) {
            std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(offset), static_cast<std::ptrdiff_t>(remaining), _partialSample.begin());
            _partialSampleSize = remaining;
            result.bytesConsumed += remaining;
            _dataBytesRemaining -= remaining;
        }

        return result;
    }

    [[nodiscard]] std::size_t maxBytesForOutputSamples(std::size_t samples) const {
        const std::size_t bytesPerSample = _format.bytesPerSample();
        if (bytesPerSample == 0U || samples == 0U) {
            return 0U;
        }

        std::size_t maxBytes = samples * bytesPerSample;
        if (_partialSampleSize > 0U) {
            maxBytes -= _partialSampleSize;
        }
        return maxBytes;
    }

    void publishFormatTag(gr::OutputSpanLike auto& outSpan, std::size_t offset) {
        auto tagMap = property_map{};
        gr::tag::put(tagMap, gr::tag::SAMPLE_RATE, sample_rate.value);
        gr::tag::put(tagMap, gr::tag::NUM_CHANNELS, channels.value);
        gr::tag::put(tagMap, gr::tag::SIGNAL_NAME, uri.value);
        outSpan.publishTag(std::move(tagMap), offset);
    }
};

static_assert(gr::BlockLike<WavSource<float>>);

} // namespace gr::audio

#endif
