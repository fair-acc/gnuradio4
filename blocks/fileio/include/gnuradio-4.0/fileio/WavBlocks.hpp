#ifndef GNURADIO_FILEIO_WAV_BLOCKS_HPP
#define GNURADIO_FILEIO_WAV_BLOCKS_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>
#include <gnuradio-4.0/fileio/FileIoTypes.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace gr::blocks::fileio {
namespace detail {

template<typename T>
concept WavSample = std::same_as<T, float> || std::same_as<T, std::int16_t>;

[[nodiscard]] inline std::uint16_t readLe16(std::span<const std::uint8_t> bytes) {
    const auto value = static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8U);
    return static_cast<std::uint16_t>(value);
}

[[nodiscard]] inline std::uint32_t readLe32(std::span<const std::uint8_t> bytes) { return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8U) | (static_cast<std::uint32_t>(bytes[2]) << 16U) | (static_cast<std::uint32_t>(bytes[3]) << 24U); }

[[nodiscard]] inline bool bytesEqual(std::span<const std::uint8_t> bytes, std::string_view text) { return bytes.size() == text.size() && std::equal(bytes.begin(), bytes.end(), text.begin(), text.end()); }

inline void writeLe16(std::uint8_t* dst, std::uint16_t value) {
    dst[0] = static_cast<std::uint8_t>(value & 0xFFU);
    dst[1] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
}

inline void writeLe32(std::uint8_t* dst, std::uint32_t value) {
    dst[0] = static_cast<std::uint8_t>(value & 0xFFU);
    dst[1] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
    dst[2] = static_cast<std::uint8_t>((value >> 16U) & 0xFFU);
    dst[3] = static_cast<std::uint8_t>((value >> 24U) & 0xFFU);
}

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

template<WavSample T>
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

template<WavSample T>
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

GR_REGISTER_BLOCK(gr::blocks::fileio::WavSource, [T], [ float, int16_t ])

template<detail::WavSample T>
struct WavSource : gr::Block<WavSource<T>> {
    using Description = Doc<R""(Streams RIFF/WAVE audio from local paths or HTTP/HTTPS URLs.

Supported WAV encodings: PCM 8/16/24/32-bit, IEEE float 32-bit, interleaved multi-channel.
Compressed formats (ADPCM, mu-law, A-law, MP3-in-WAV) are not supported.)"">;

    gr::PortOut<T> out;

    gr::Annotated<std::string, "uri", gr::Visible, gr::Doc<"Local file path or HTTP/HTTPS URL to a WAV file">>                                uri;
    gr::Annotated<Mode, "mode", gr::Visible, gr::Doc<"overwrite/append: single file, multi: read all matching files in order">>               mode         = Mode::overwrite;
    gr::Annotated<bool, "repeat", gr::Doc<"true: restart from beginning when all files are exhausted">>                                       repeat       = false;
    gr::Annotated<gr::Size_t, "offset", gr::Visible, gr::Doc<"Skip this many samples at the start of each file's data chunk">>                offset       = 0U;
    gr::Annotated<gr::Size_t, "length", gr::Visible, gr::Doc<"Max samples to read per file (0 = read entire file)">>                          length       = 0U;
    gr::Annotated<std::string, "trigger_name", gr::Doc<"Trigger name tag emitted at the start of each file">>                                 trigger_name = std::string("WavSource::start");
    gr::Annotated<float, "sample_rate", gr::Visible, gr::Unit<"Hz">, gr::Doc<"Decoded sample rate (read-only, updated from WAV header)">>     sample_rate  = 0.f;
    gr::Annotated<gr::Size_t, "num_channels", gr::Visible, gr::Doc<"Decoded interleaved channel count (read-only, updated from WAV header)">> num_channels = 0U;

    GR_MAKE_REFLECTABLE(WavSource, out, uri, mode, repeat, offset, length, trigger_name, sample_rate, num_channels);

    static constexpr std::size_t kMaxHeaderBytes = 1024U * 1024U;

    struct DecodeResult {
        std::size_t samplesWritten{0U};
        std::size_t bytesConsumed{0U};
    };

    gr::algorithm::fileio::Reader      _reader;
    std::vector<std::filesystem::path> _filesToRead;
    std::size_t                        _currentFileIndex{0U};
    bool                               _headerParsed{false};
    bool                               _readerFinalSeen{false};
    bool                               _readerActive{false};
    bool                               _failed{false};
    detail::WavFormat                  _format{};
    std::size_t                        _dataBytesRemaining{0U};
    std::size_t                        _totalSamplesRead{0U};
    std::size_t                        _offsetBytesRemaining{0U};
    std::vector<std::uint8_t>          _headerBuffer;
    std::array<std::uint8_t, 4>        _partialSample{};
    std::size_t                        _partialSampleSize{0U};
    bool                               _formatTagPending{true};

    using gr::Block<WavSource<T>>::Block;

    void resetFileState() {
        _reader          = {};
        _headerParsed    = false;
        _readerFinalSeen = false;
        _readerActive    = false;
        _failed          = false;
        _format          = {};
        _headerBuffer.clear();
        _dataBytesRemaining   = 0U;
        _totalSamplesRead     = 0U;
        _offsetBytesRemaining = 0U;
        _partialSampleSize    = 0U;
        _formatTagPending     = true;
    }

    void fail(std::string_view endpoint, gr::Error error) {
        this->emitErrorMessage(endpoint, error);
        _reader.cancel();
        _readerActive    = false;
        _readerFinalSeen = true;
        _failed          = true;
    }

    void start() {
        _currentFileIndex = 0U;
        sample_rate       = 0.f;
        num_channels      = 0U;
        _filesToRead.clear();

        std::filesystem::path filePath(uri.value);
        if (mode.value == Mode::multi) {
            if (std::filesystem::exists(filePath.parent_path())) {
                auto stem = filePath.filename().string();
                for (const auto& entry : std::filesystem::directory_iterator(filePath.parent_path())) {
                    if (entry.is_regular_file() && entry.path().string().find(stem) != std::string::npos) {
                        _filesToRead.push_back(entry.path());
                    }
                }
                std::sort(_filesToRead.begin(), _filesToRead.end());
            }
        } else {
            _filesToRead.push_back(filePath);
        }

        openNextFile();
    }

    void stop() {
        _reader.cancel();
        resetFileState();
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

        // skip offset bytes from the header buffer crossover
        if (_headerParsed && _offsetBytesRemaining > 0U && !_headerBuffer.empty()) {
            const std::size_t toSkip = std::min(_offsetBytesRemaining, _headerBuffer.size());
            _headerBuffer.erase(_headerBuffer.begin(), _headerBuffer.begin() + static_cast<std::ptrdiff_t>(toSkip));
            _offsetBytesRemaining -= toSkip;
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

        if (_headerParsed && _dataBytesRemaining == 0U && _partialSampleSize > 0U) {
            fail("WavSource::processBulk()", gr::Error("WAV data ended mid-sample"));
            outSpan.publish(0U);
            return gr::work::Status::ERROR;
        }

        if (published > 0U && _formatTagPending) {
            publishFormatTag(outSpan, 0U);
            _formatTagPending = false;
        }

        outSpan.publish(published);

        if (requiredOutputSize && published == 0U) {
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        const bool lengthExhausted = length.value != 0U && _totalSamplesRead >= static_cast<std::size_t>(length.value);
        if (_headerParsed && (_dataBytesRemaining == 0U || lengthExhausted)) {
            return finishCurrentFile();
        }

        return gr::work::Status::OK;
    }

private:
    void openNextFile() {
        if (_currentFileIndex >= _filesToRead.size()) {
            return;
        }
        resetFileState();

        gr::algorithm::fileio::ReaderConfig config{};
        config.chunkBytes          = std::max<std::size_t>(1U, out.max_buffer_size() * sizeof(T) / 4U);
        config.chunkAlignmentBytes = 1U;

        auto readerExp = gr::algorithm::fileio::readAsync(_filesToRead[_currentFileIndex].string(), std::move(config));
        if (!readerExp) {
            fail("WavSource::openNextFile()", readerExp.error());
            return;
        }
        _reader       = std::move(readerExp.value());
        _readerActive = true;
        _currentFileIndex++;
    }

    [[nodiscard]] gr::work::Status finishCurrentFile() {
        _reader.cancel();
        _readerActive = false;
        if (_currentFileIndex < _filesToRead.size()) {
            openNextFile();
            return gr::work::Status::OK;
        }
        if (repeat && !_filesToRead.empty()) {
            _currentFileIndex = 0U;
            openNextFile();
            return gr::work::Status::OK;
        }
        return gr::work::Status::DONE;
    }

    [[nodiscard]] std::expected<bool, gr::Error> tryParseHeader() {
        const auto bytes = std::span<const std::uint8_t>(_headerBuffer);

        if (bytes.size() < 12U) {
            return false;
        }

        if (!detail::bytesEqual(bytes.subspan(0U, 4U), "RIFF") || !detail::bytesEqual(bytes.subspan(8U, 4U), "WAVE")) {
            return std::unexpected(gr::Error("Not a valid WAVE file"));
        }

        std::size_t pos     = 12U;
        bool        haveFmt = false;

        while (pos + 8U <= bytes.size()) {
            const auto chunkId   = bytes.subspan(pos, 4U);
            const auto chunkSize = static_cast<std::size_t>(detail::readLe32(bytes.subspan(pos + 4U, 4U)));
            pos += 8U;

            if (detail::bytesEqual(chunkId, "data")) {
                if (!haveFmt) {
                    return std::unexpected(gr::Error("WAV data chunk before fmt chunk"));
                }
                if ((chunkSize % _format.blockAlign) != 0U) {
                    return std::unexpected(gr::Error("WAV data chunk size is not a multiple of block_align"));
                }
                _dataBytesRemaining = chunkSize;

                // apply sample offset
                const std::size_t offsetBytes = static_cast<std::size_t>(offset.value) * _format.bytesPerSample();
                if (offsetBytes > 0U && offsetBytes < _dataBytesRemaining) {
                    _offsetBytesRemaining = offsetBytes;
                    _dataBytesRemaining -= offsetBytes;
                } else if (offsetBytes >= _dataBytesRemaining) {
                    _dataBytesRemaining = 0U;
                }

                // apply length limit
                if (length.value != 0U) {
                    const std::size_t maxBytes = static_cast<std::size_t>(length.value) * _format.bytesPerSample();
                    _dataBytesRemaining        = std::min(_dataBytesRemaining, maxBytes);
                }

                if (pos < bytes.size()) {
                    std::vector<std::uint8_t> crossover(bytes.begin() + static_cast<std::ptrdiff_t>(pos), bytes.end());
                    _headerBuffer = std::move(crossover);
                } else {
                    _headerBuffer.clear();
                }

                return true;
            }

            const std::size_t paddedSize = chunkSize + (chunkSize & 1U);
            if (pos + paddedSize > bytes.size()) {
                return false;
            }

            if (detail::bytesEqual(chunkId, "fmt ")) {
                auto fmtExp = detail::parseWavFmtChunk(bytes.subspan(pos, chunkSize));
                if (!fmtExp) {
                    return std::unexpected(fmtExp.error());
                }
                _format      = fmtExp.value();
                sample_rate  = static_cast<float>(_format.sampleRate);
                num_channels = static_cast<gr::Size_t>(_format.channels);
                haveFmt      = true;
            }

            pos += paddedSize;
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

        std::size_t pos = 0U;

        if (_partialSampleSize > 0U) {
            const std::size_t needed    = bytesPerSample - _partialSampleSize;
            const std::size_t available = std::min(needed, bytes.size());

            std::copy_n(bytes.begin(), static_cast<std::ptrdiff_t>(available), _partialSample.begin() + static_cast<std::ptrdiff_t>(_partialSampleSize));
            _partialSampleSize += available;
            pos += available;
            result.bytesConsumed += available;
            _dataBytesRemaining -= available;

            if (_partialSampleSize == bytesPerSample) {
                output[result.samplesWritten++] = detail::decodeWavSample<T>(std::span(_partialSample).first(bytesPerSample), _format);
                _partialSampleSize              = 0U;
            } else {
                return result;
            }
        }

        const std::size_t maxSamples = std::min(output.size() - result.samplesWritten, (bytes.size() - pos) / bytesPerSample);
        for (std::size_t i = 0U; i < maxSamples; ++i) {
            output[result.samplesWritten++] = detail::decodeWavSample<T>(bytes.subspan(pos, bytesPerSample), _format);
            pos += bytesPerSample;
            result.bytesConsumed += bytesPerSample;
            _dataBytesRemaining -= bytesPerSample;
        }

        const std::size_t remaining = bytes.size() - pos;
        if (remaining > 0U && remaining < bytesPerSample) {
            std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(pos), static_cast<std::ptrdiff_t>(remaining), _partialSample.begin());
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

    void publishFormatTag(gr::OutputSpanLike auto& outSpan, std::size_t tagOffset) {
        auto tagMap = property_map{};
        gr::tag::put(tagMap, gr::tag::SAMPLE_RATE, sample_rate.value);
        gr::tag::put(tagMap, gr::tag::NUM_CHANNELS, num_channels.value);
        gr::tag::put(tagMap, gr::tag::SIGNAL_NAME, uri.value);
        if (!trigger_name.value.empty()) {
            gr::tag::put(tagMap, gr::tag::TRIGGER_NAME, trigger_name.value);
            gr::tag::put(tagMap, gr::tag::TRIGGER_TIME, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
            gr::tag::put(tagMap, gr::tag::TRIGGER_OFFSET, 0.f);
        }
        outSpan.publishTag(std::move(tagMap), tagOffset);
    }
};

static_assert(gr::BlockLike<WavSource<float>>);

GR_REGISTER_BLOCK(gr::blocks::fileio::WavSink, [T], [ float, int16_t ])

template<detail::WavSample T>
struct WavSink : gr::Block<WavSink<T>> {
    using Description = Doc<R""(Writes interleaved PCM samples to a RIFF/WAVE file.

`WavSink<float>` writes IEEE float32 (format tag 3). `WavSink<int16_t>` writes PCM 16-bit
(format tag 1). Multi-channel interleaved audio is supported for any channel count.
In multi mode, rotates to a new timestamped file when max_bytes_per_file is reached.)"">;

    gr::PortIn<T> in;

    gr::Annotated<std::string, "uri", gr::Visible, gr::Doc<"Output file path">>                                                               uri;
    gr::Annotated<Mode, "mode", gr::Visible, gr::Doc<"overwrite: truncate, multi: rotate to new timestamped file on max_bytes_per_file">>     mode                  = Mode::overwrite;
    gr::Annotated<gr::Size_t, "max_bytes_per_file", gr::Visible, gr::Doc<"Max bytes per file before rotating in multi mode (0 = unlimited)">> max_bytes_per_file    = 0U;
    gr::Annotated<float, "sample_rate", gr::Visible, gr::Unit<"Hz">, gr::Doc<"Sample rate for the WAV header">>                               sample_rate           = 48000.f;
    gr::Annotated<gr::Size_t, "num_channels", gr::Visible, gr::Doc<"Interleaved channel count for the WAV header">>                           num_channels          = 1U;
    gr::Annotated<gr::Size_t, "total_samples_written", gr::Doc<"Read-only: total interleaved samples written across all files">>              total_samples_written = 0U;

    GR_MAKE_REFLECTABLE(WavSink, in, uri, mode, max_bytes_per_file, sample_rate, num_channels, total_samples_written);

    static constexpr std::size_t   kHeaderSize  = 44U;
    static constexpr std::uint16_t kFormatTag   = std::same_as<T, float> ? std::uint16_t(3U) : std::uint16_t(1U);
    static constexpr std::uint16_t kBitsPerSamp = std::same_as<T, float> ? std::uint16_t(32U) : std::uint16_t(16U);

    std::ofstream _file;
    bool          _headerWritten{false};
    std::size_t   _fileSamplesWritten{0U};
    std::size_t   _fileCounter{0U};

    using gr::Block<WavSink<T>>::Block;

    void start() {
        total_samples_written = 0U;
        _fileCounter          = 0U;
        openNextFile();
    }

    void stop() { closeFile(); }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& inSpan) {
        if (!_file.is_open()) {
            std::ignore = inSpan.consume(0U);
            return gr::work::Status::ERROR;
        }

        if (!_headerWritten) {
            writeHeader();
            _headerWritten = true;
        }

        // rotate in multi mode
        if (mode.value == Mode::multi && max_bytes_per_file.value != 0U && _fileSamplesWritten * sizeof(T) >= max_bytes_per_file.value) {
            closeFile();
            openNextFile();
            if (!_file.is_open()) {
                std::ignore = inSpan.consume(0U);
                return gr::work::Status::ERROR;
            }
            writeHeader();
            _headerWritten = true;
        }

        std::size_t n = inSpan.size();
        if (n == 0U) {
            std::ignore = inSpan.consume(0U);
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        // cap to max_bytes_per_file in multi mode
        if (mode.value == Mode::multi && max_bytes_per_file.value != 0U) {
            const std::size_t remaining = (max_bytes_per_file.value - _fileSamplesWritten * sizeof(T)) / sizeof(T);
            n                           = std::min(n, remaining);
        }

        _file.write(reinterpret_cast<const char*>(inSpan.data()), static_cast<std::streamsize>(n * sizeof(T)));
        _fileSamplesWritten += n;
        total_samples_written = total_samples_written.value + static_cast<gr::Size_t>(n);

        std::ignore = inSpan.consume(n);
        return gr::work::Status::OK;
    }

private:
    void openNextFile() {
        _headerWritten      = false;
        _fileSamplesWritten = 0U;

        if (uri.value.empty()) {
            this->emitErrorMessage("WavSink::start()", gr::Error("uri is empty"));
            return;
        }

        std::string actualPath;
        if (mode.value == Mode::multi) {
            std::filesystem::path filePath(uri.value);
            const auto            now   = std::chrono::system_clock::now();
            const auto            timeT = std::chrono::system_clock::to_time_t(now);
            std::array<char, 32>  timeBuf{};
            std::strftime(timeBuf.data(), timeBuf.size(), "%Y%m%dT%H%M%S", std::gmtime(&timeT));
            actualPath = (filePath.parent_path() / std::format("{}_{}_{}", std::string_view(timeBuf.data()), _fileCounter++, filePath.filename().string())).string();
            std::filesystem::create_directories(std::filesystem::path(actualPath).parent_path());
        } else {
            actualPath = uri.value;
            std::filesystem::create_directories(std::filesystem::path(actualPath).parent_path());
        }

        _file.open(actualPath, std::ios::binary | std::ios::trunc);
        if (!_file.is_open()) {
            this->emitErrorMessage("WavSink::openNextFile()", gr::Error(std::format("cannot open '{}' for writing", actualPath)));
        }
    }

    void closeFile() {
        if (_file.is_open() && _headerWritten) {
            patchHeaderSizes();
        }
        _file.close();
    }

    void writeHeader() {
        const auto ch         = static_cast<std::uint16_t>(std::max(1U, num_channels.value));
        const auto rate       = static_cast<std::uint32_t>(std::max(1.f, sample_rate.value));
        const auto blockAlign = static_cast<std::uint16_t>(ch * (kBitsPerSamp / 8U));
        const auto byteRate   = static_cast<std::uint32_t>(rate) * static_cast<std::uint32_t>(blockAlign);

        std::array<std::uint8_t, kHeaderSize> hdr{};
        std::copy_n("RIFF", 4, hdr.begin());
        detail::writeLe32(hdr.data() + 4U, 0U); // patched on close
        std::copy_n("WAVE", 4, hdr.begin() + 8);
        std::copy_n("fmt ", 4, hdr.begin() + 12);
        detail::writeLe32(hdr.data() + 16U, 16U);
        detail::writeLe16(hdr.data() + 20U, kFormatTag);
        detail::writeLe16(hdr.data() + 22U, ch);
        detail::writeLe32(hdr.data() + 24U, rate);
        detail::writeLe32(hdr.data() + 28U, byteRate);
        detail::writeLe16(hdr.data() + 32U, blockAlign);
        detail::writeLe16(hdr.data() + 34U, kBitsPerSamp);
        std::copy_n("data", 4, hdr.begin() + 36);
        detail::writeLe32(hdr.data() + 40U, 0U); // patched on close

        _file.write(reinterpret_cast<const char*>(hdr.data()), static_cast<std::streamsize>(kHeaderSize));
    }

    void patchHeaderSizes() {
        const auto dataBytes = static_cast<std::uint32_t>(_fileSamplesWritten * sizeof(T));
        const auto riffSize  = static_cast<std::uint32_t>(kHeaderSize - 8U + dataBytes);

        std::array<std::uint8_t, 4> buf{};
        detail::writeLe32(buf.data(), riffSize);
        _file.seekp(4);
        _file.write(reinterpret_cast<const char*>(buf.data()), 4);

        detail::writeLe32(buf.data(), dataBytes);
        _file.seekp(40);
        _file.write(reinterpret_cast<const char*>(buf.data()), 4);
    }
};

static_assert(gr::BlockLike<WavSink<float>>);

} // namespace gr::blocks::fileio

#endif
