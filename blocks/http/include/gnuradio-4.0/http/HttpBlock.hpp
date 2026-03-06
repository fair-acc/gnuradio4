#ifndef GNURADIO_HTTP_BLOCK_HPP
#define GNURADIO_HTTP_BLOCK_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <format>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifdef __GNUC__
#pragma GCC diagnostic push
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

using namespace gr;

namespace gr::http {

namespace fileio = gr::algorithm::fileio;

enum class SourceMode : char {
    GET       = 1,
    SUBSCRIBE = 2,
};

GR_REGISTER_BLOCK(gr::http::HttpSource)
struct HttpSource : Block<HttpSource> {
    using Description = Doc<R""(
Read data from an HTTP endpoint.

GET reads one response.
SUBSCRIBE keeps polling and publishes each new response.

Each output item is a PMT map with:
- status: HTTP status code
- raw-data: response body bytes
- mime-type: response content type

Internally this uses FileIo.
)"">;

    PortOut<pmt::Value::Map> out;

    gr::Annotated<std::string, "URI">                                                       url;
    gr::Annotated<gr::http::SourceMode, "type", gr::Doc<"GET, SUBSCRIBE">>                  type        = gr::http::SourceMode::GET;
    gr::Annotated<gr::Size_t, "chunk_bytes", gr::Doc<"Chunk size in bytes, 0 = no limits">> chunk_bytes = 0U;

    GR_MAKE_REFLECTABLE(HttpSource, out, url, type, chunk_bytes);

    fileio::Reader _reader;
    bool           _emscriptenRunOnMainThread = true; // used in Emscripten unit-tests only

    [[nodiscard]] static pmt::Value::Map makeResultValue(std::span<const std::uint8_t> rawData, int status = 200, std::string_view mimeType = "text/plain") {
        pmt::Value::Map result;
        result["mime-type"] = std::string(mimeType);
        result["status"]    = status;
        result["raw-data"]  = gr::Tensor<std::uint8_t>(rawData.begin(), rawData.end());
        return result;
    }

    [[nodiscard]] fileio::ReaderConfig readerConfig() const {
        fileio::ReaderConfig config;
        if (chunk_bytes.value != 0U) {
            config.chunkBytes = static_cast<std::size_t>(chunk_bytes.value);
        }
        config.longPolling               = type.value == SourceMode::SUBSCRIBE;
        config.emscriptenRunOnMainThread = _emscriptenRunOnMainThread;
        return config;
    }

    void openReader() {
        auto readerExp = fileio::readAsync(url.value, readerConfig());
        if (!readerExp.has_value()) {
            throw gr::exception(readerExp.error().message, readerExp.error().sourceLocation);
        }
        _reader = std::move(readerExp.value());
    }

    void start() { openReader(); }

    void stop() { _reader.cancel(); }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (lifecycle::isActive(this->state()) && (newSettings.contains("url") || newSettings.contains("type") || newSettings.contains("chunk_bytes"))) {
            _reader.cancel();
            openReader();
        }
    }

    [[nodiscard]] work::Status processBulk(OutputSpanLike auto& outSpan) {
        if (outSpan.empty()) {
            return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        bool                           finished = false;
        std::optional<pmt::Value::Map> result;
        std::optional<gr::Error>       error;
        std::size_t                    nSamplesToPublish = 0U;

        _reader.poll(
            [&](const auto& res) {
                finished = res.isFinal;

                if (res.data.has_value()) {
                    const auto bytes = res.data.value();
                    if (!bytes.empty()) {
                        result            = makeResultValue(bytes);
                        nSamplesToPublish = 1U;
                    }
                    return;
                }
                error = res.data.error();
            },
            std::numeric_limits<std::size_t>::max(), false);

        if (error.has_value()) {
            throw gr::exception(error->message, error->sourceLocation);
        }
        if (result.has_value()) {
            outSpan[0] = std::move(*result);
        }
        outSpan.publish(nSamplesToPublish);
        if (finished) {
            return work::Status::DONE;
        }
        return work::Status::OK;
    }
};

GR_REGISTER_BLOCK(gr::http::HttpSink)
struct HttpSink : Block<HttpSink> {
    using Description = Doc<R""(
Send incoming bytes to an HTTP endpoint with POST.

Each input chunk is sent as one POST request.
Use content_type to set the Content-Type header.

Internally this uses FileIo.
)"">;

    PortIn<std::uint8_t> in;

    gr::Annotated<std::string, "URI">                                               url;
    gr::Annotated<std::string, "content_type", gr::Doc<"HTTP Content-Type header">> content_type = "application/octet-stream";

    GR_MAKE_REFLECTABLE(HttpSink, in, url, content_type);

    std::optional<fileio::Writer> _writer;
    bool                          _emscriptenRunOnMainThread = true; // used in Emscripten unit-tests only

    [[nodiscard]] fileio::WriterConfig writerConfig() const {
        fileio::WriterConfig config;
        if (!content_type.value.empty()) {
            config.httpHeaders.emplace("Content-Type", content_type.value);
        }
        config.emscriptenRunOnMainThread = _emscriptenRunOnMainThread;
        return config;
    }

    void start() { _writer.reset(); }

    void stop() {
        if (_writer.has_value()) {
            _writer->cancel();
        }
    }

    [[nodiscard]] work::Status processBulk(InputSpanLike auto& inSpan) {
        if (inSpan.empty()) {
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        if (_writer.has_value()) {
            if (!_writer->finished()) {
                return work::Status::OK;
            }

            auto writeResultExp = _writer->result();
            _writer.reset();
            if (!writeResultExp.has_value()) {
                throw gr::exception(writeResultExp.error().message, writeResultExp.error().sourceLocation);
            }
        }

        auto bytes     = std::span<const std::uint8_t>(inSpan.data(), inSpan.size());
        auto writerExp = fileio::writeAsync(url.value, bytes, writerConfig());
        if (!writerExp.has_value()) {
            throw gr::exception(writerExp.error().message, writerExp.error().sourceLocation);
        }
        _writer = std::move(writerExp.value());

        return work::Status::OK;
    }
};

} // namespace gr::http

#endif // GNURADIO_HTTP_BLOCK_HPP
