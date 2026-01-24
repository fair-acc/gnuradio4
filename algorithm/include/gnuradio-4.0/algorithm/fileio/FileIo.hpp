#ifndef FILEIO_HPP
#define FILEIO_HPP

#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIoEmscriptenHelper.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIoHelpers.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if !defined(__EMSCRIPTEN__) && GR_HTTP_ENABLED
#include <cpr/cpr.h>
#endif
#if __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/emscripten.h>
#include <emscripten/fetch.h>
#include <emscripten/html5.h>
#include <emscripten/threading.h>
#endif

/*  File I/O helpers for gnuradio.
 *
 *  It implements a unified Reader/Writer abstractions for:
 *    - Local files via file:/ URIs (native + MEMFS under Emscripten)
 *    - HTTP(S) GET / long-polling via http:// and https://
 *    - In-memory sources (readAsync(span<const uint8_t>))
 *    - dialog:/ URIs (open file dialog -> memory or local file)
 *    - download:/ URIs (WASM "download to disk")
 *
 *  Reader:
 *    - ReaderConfig controls chunkBytes, offset, httpTimeoutNanos and longPolling.
 *    - readAsync(uri, config) returns a Reader backed by a CircularBuffer.
 *    - poll(cb, maxSize, doWait) delivers chunks or errors via PollResult, where cb is called with either data or an error, and a final message.
 *      maxSize should typically be set to number of free bytes in the caller’s output buffer. This guarantees that all data reported by a single poll()
 *      call can be written directly without intermediate storage. The default is std::numeric_limits<std::size_t>::max(), meaning "no explicit limit".
 *      The caller should ensure that ReaderConfig::chunkBytes <= maxSize; otherwise an error is returned, and if that error is not handled correctly in cb,
 *      user code may end up in an infinite loop.
 *    - get() is a convenience: it blocks on a worker thread and concatenates all data into a single std::vector<uint8_t>.
 *    - cancel() requests a soft cancel; for Emscripten this is best-effort (we cannot synchronously abort emscripten_fetch on another thread).
 *
 *    Example (local file):
 *      using namespace gr::algorithm::fileio;
 *      auto rExp = readAsync("file:/tmp/test.bin", {.chunkBytes = 4096});
 *      if (!rExp) throw std::runtime_error(rExp.error().message);
 *       // collect whole file
 *      auto dataExp = rExp->get();
 *      // OR poll individual chunks
 *      rExp->poll([](const Reader::PollResult& res) {
 *          if (res.data) {
 *              // process res.data.value()
 *          } else {
 *              std::println("Error: {}", res.data.error().message);
 *          }
 *      }, maxSize, doWait);
 *
 *  Writer side:
 *    - WriterConfig supports WriteMode::overwrite and WriteMode::append (append is only valid for local files, not HTTP).
 *    - writeAsync(uri, bytes, config) runs on a worker pool and returns a Writer with wait() + result().
 *    - write(uri, bytes, config) is a synchronous convenience wrapper.
 *    - For local files, WriteResult always has httpStatus=0 and an empty body.
 *
 *  Native HTTP implementation:
 *    - Enabled when GR_HTTP_ENABLED=1, using CPR::Session.
 *    - Long-polling is handled by repeatedly calling runHttpGetNativeOnce().
 *    - HTTP errors (status >= 400) or CPR errors are reported as gr::Error.
 *
 *  CMake controls for native HTTP (CPR + libcurl):
 *    - GR_ENABLE_HTTP=ON: Require libcurl. If libcurl is found, CPR is enabled and GR_HTTP_ENABLED=1; otherwise CMake fails with an error.
 *    - GR_ENABLE_HTTP=OPTIONAL: Enable CPR only if system libcurl is found. If libcurl is missing, GR_HTTP_ENABLED=0 and HTTP(S) operations are treated as "disabled at build time".
 *    - GR_ENABLE_HTTP = OFF: CPR is never fetched; GR_HTTP_ENABLED=0 and all HTTP(S) operations return "disabled at build time".
 *
 *  Emscripten implementation:
 *    - Uses emscripten_fetch + callbacks for GET/POST, with optional longPolling (recursive re-fetch).
 *    - Blocking wait() / write() is explicitly forbidden on the main WASM runtime thread; attempts are detected and return an error/warning.
 *    - download:/ URIs trigger a JS "download" in the browser.
 */

namespace gr::algorithm::fileio {

// Read from file and http GET

namespace detail {
inline constexpr std::string_view kMessageDataKey = "data";
}

struct ReaderConfig {
    // Note about chunkBytes: When using Reader::poll(cb, maxSize, doWait) - the caller should ensure that chunkBytes <= maxSize; otherwise an error is returned,
    // and if that error is not handled correctly in cb, user code may end up in an infinite loop.
    std::size_t                        chunkBytes       = std::numeric_limits<std::size_t>::max();
    std::optional<std::size_t>         offset           = std::nullopt;
    std::uint64_t                      httpTimeoutNanos = 30'000'000'000; // 30 s time-out for http(s)
    std::size_t                        bufferMinSize    = 1024;           // one gr::Message per slot
    bool                               longPolling      = false;
    std::map<std::string, std::string> httpHeaders      = {};
    bool                               tlsVerifyPeer    = true; // for native https
};

struct ReaderState {
    std::string  uri;
    ReaderConfig config;

    gr::CircularBuffer<gr::Message> buffer;
    decltype(buffer.new_writer())   bufferWriter;
    decltype(buffer.new_reader())   bufferReader;

    std::atomic<std::size_t> updateCounter{0};
    std::atomic<bool>        cancelRequested{false};
    std::atomic<bool>        finalPublished{false}; // mostly for debugging

    std::unique_ptr<DialogOpenHandle> dialogHandle; // Present only for dialog:/ readers

    ReaderState(std::string uri_, ReaderConfig config_) //
        : uri(std::move(uri_)), config(config_), buffer(config_.bufferMinSize), bufferWriter(buffer.new_writer()), bufferReader(buffer.new_reader()) {}

#ifndef NDEBUG
    ~ReaderState() {
        if (!finalPublished.load(std::memory_order_acquire)) {
            std::puts("ReaderState destroyed without final message");
        }
    }
#endif
};

struct Reader {

    struct PollResult {
        std::expected<std::span<const std::uint8_t>, gr::Error> data = std::span<const std::uint8_t>{}; // span is valid only during callback
        bool                                                    isFinal{false};
    };

private:
    std::shared_ptr<ReaderState> _state;

public:
    Reader() = default;
    explicit Reader(std::shared_ptr<ReaderState> state) : _state(std::move(state)) {}

    [[nodiscard]] bool cancelRequested() const { return _state && _state->cancelRequested.load(std::memory_order_acquire); }

    // Note for Emscripten: we do a "soft" cancel. emscripten_fetch_close() is not reliably
    // usable from another thread, so cancellation is best-effort and not immediate.
    // A final message (either success or error) will always be published.
    void cancel() {
        if (_state != nullptr) {
            _state->cancelRequested.store(true, std::memory_order_release);
            _state->updateCounter.fetch_add(1, std::memory_order_release);
            _state->updateCounter.notify_all();
        }
    }

    void wait() {
        if (_state == nullptr) {
            return;
        }

#if __EMSCRIPTEN__
        // Do NOT block main WASM thread.
        if (emscripten_is_main_runtime_thread()) {
            std::puts("fileio wait() on main WASM thread is not allowed; returning immediately");
            return;
        }
#endif

        while (true) {
            if (_state->bufferReader.available() > 0) {
                return;
            }
            if (_state->cancelRequested.load(std::memory_order_acquire)) {
                return;
            }
            auto expected = _state->updateCounter.load(std::memory_order_acquire);
            _state->updateCounter.wait(expected, std::memory_order_acquire);
        }
    }

    // Note: The caller should ensure that ReaderConfig::chunkBytes <= maxSize; otherwise an error is returned,
    // and if that error is not handled correctly in callback, user code may end up in an infinite loop.
    template<typename TCallback>
    requires std::invocable<TCallback, PollResult> && std::same_as<std::invoke_result_t<TCallback, PollResult>, void>
    void poll(TCallback&& callback, std::size_t maxSize = std::numeric_limits<std::size_t>::max(), bool doWait = false) {
        if (doWait) {
            wait();
        }

        if (_state == nullptr || _state->bufferReader.available() == 0) {
            return;
        }

        auto span = _state->bufferReader.get(1);
        if (span.empty()) {
            return;
        }

        auto&      msg = span[0];
        PollResult res;
        res.isFinal = (msg.cmd == gr::message::Command::Final);

        if (!msg.data.has_value()) {
            res.data = std::unexpected(std::move(msg.data.error()));
            std::invoke(std::forward<TCallback>(callback), std::move(res));
            std::ignore = span.consume(1);
            return;
        }

        auto it = msg.data->find(detail::kMessageDataKey);
        if (it != msg.data->end()) {
            if (auto bytes = it->second.get_if<Tensor<std::uint8_t>>(); bytes != nullptr) {
                if (bytes->size() <= maxSize) {
                    res.data = std::span<const std::uint8_t>(bytes->data(), bytes->size());
                    std::invoke(std::forward<TCallback>(callback), std::move(res));
                    std::ignore = span.consume(1);
                    return;
                } else {
                    res.data = std::unexpected{gr::Error{std::format("message data size ({}) exceeds max requested size ({})", bytes->size(), maxSize)}};
                    std::invoke(std::forward<TCallback>(callback), std::move(res));
                    std::ignore = span.consume(0);
                    return;
                }
            }
        }

        std::invoke(std::forward<TCallback>(callback), std::move(res));
        std::ignore = span.consume(1);
    }

    std::expected<std::vector<std::uint8_t>, gr::Error> get() {
        bool                      finished = false;
        std::vector<std::uint8_t> allData;
        std::optional<gr::Error>  firstError;
        while (!finished) {
            poll(
                [&finished, &allData, &firstError](const auto& res) {
                    finished = res.isFinal;
                    if (res.data.has_value()) {
                        auto data = res.data.value();
                        if (!data.empty()) {
                            allData.insert(allData.end(), data.begin(), data.end());
                        }
                    } else {
                        const auto& err = res.data.error();
                        if (!firstError) {
                            firstError = err;
                        }
                    }
                },
                std::numeric_limits<std::size_t>::max(), true);
        }
        if (firstError) {
            return std::unexpected(*firstError);
        }
        return allData;
    }
};

// `push*` methods take raw pointer to `ReadState` to satisfy Emscripten’s interface and unify the API across implementations.
inline void publishMessage(ReaderState* state, gr::Message&& m) {
    if (state == nullptr) {
        return;
    }

    {
        auto span = state->bufferWriter.template reserve<gr::SpanReleasePolicy::ProcessNone>(1);
        span[0]   = std::move(m);
        span.publish(1);
    }
    state->updateCounter.fetch_add(1, std::memory_order_release);
    state->updateCounter.notify_all();
}

inline void pushData(ReaderState* state, Tensor<std::uint8_t>&& values) {
    if (state == nullptr || state->cancelRequested.load(std::memory_order_acquire) || values.empty()) {
        return;
    }

    gr::Message msg;
    msg.endpoint = state->uri;
    msg.cmd      = gr::message::Command::Notify;
    msg.data     = gr::property_map{{std::pmr::string(detail::kMessageDataKey), std::move(values)}};
    publishMessage(state, std::move(msg));
}

inline void pushData(ReaderState* state, std::span<const std::uint8_t> data) {
    if (state == nullptr || data.empty() || state->cancelRequested.load(std::memory_order_acquire)) {
        return;
    }

    const std::size_t total = data.size();
    const std::size_t chunk = std::max<std::size_t>(1, state->config.chunkBytes);
    std::size_t       pos   = 0;
    while (pos < total) {
        const std::size_t n     = std::min(chunk, total - pos);
        auto              first = std::next(data.begin(), static_cast<std::ptrdiff_t>(pos));
        auto              last  = std::next(first, static_cast<std::ptrdiff_t>(n));
        pushData(state, Tensor<std::uint8_t>(first, last));
        pos += n;
    }
}

inline void pushError(ReaderState* state, std::string_view msg) {
    if (state == nullptr) {
        return;
    }
    gr::Message errorMsg;
    errorMsg.endpoint = state->uri;
    errorMsg.cmd      = gr::message::Command::Notify;
    errorMsg.data     = std::unexpected(gr::Error{msg});
    publishMessage(state, std::move(errorMsg));
}

inline void pushFinal(ReaderState* state) {
    if (state == nullptr) {
        return;
    }

    bool expected = false;
    if (!state->finalPublished.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        throw gr::exception("pushFinal called more than once on the same ReadState");
    }

    gr::Message finalMsg;
    finalMsg.endpoint = state->uri;
    finalMsg.cmd      = gr::message::Command::Final;
    finalMsg.data     = gr::property_map{};
    publishMessage(state, std::move(finalMsg));
}

inline void pushErrorFinal(ReaderState* state, std::string_view msg) {
    pushError(state, msg);
    pushFinal(state);
}

inline void runReadMemorySource(ReaderState* state, std::span<const std::uint8_t> bytes) {
    if (state == nullptr) {
        return;
    }

    const std::size_t offset = state->config.offset.value_or(0);
    if (offset >= bytes.size()) {
        pushFinal(state);
        return;
    }

    pushData(state, bytes.subspan(offset));
    pushFinal(state);
}

inline void runReadMemorySource(std::shared_ptr<ReaderState> state, std::span<const std::uint8_t> bytes) { runReadMemorySource(state.get(), bytes); }

inline void runReadLocalFile(std::shared_ptr<ReaderState> state) {
    namespace fs = std::filesystem;

    if (state == nullptr) {
        return;
    }

    fs::path path = state->uri;

    std::error_code ec;
    if (!fs::exists(path, ec) || !fs::is_regular_file(path, ec)) {
        pushErrorFinal(state.get(), std::format("file not found or not a regular file: {}", state->uri));
        return;
    }

    const auto fileSize = fs::file_size(path, ec);
    if (ec) {
        pushErrorFinal(state.get(), std::format("failed to get file size: {} (ec={})", state->uri, ec.value()));
        return;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        pushErrorFinal(state.get(), std::format("failed to open file: {}", state->uri));
        return;
    }

    const std::size_t offset = state->config.offset.value_or(0);
    if (offset >= fileSize) {
        pushFinal(state.get());
        return;
    }
    if (offset > 0) {
        in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!in) {
            pushErrorFinal(state.get(), std::format("seek failed to offset {} in file: {}", offset, state->uri));
            return;
        }
    }

    const std::size_t chunk     = std::max<std::size_t>(1, state->config.chunkBytes);
    std::size_t       remaining = static_cast<std::size_t>(fileSize) - offset;
    while (!state->cancelRequested.load(std::memory_order_acquire)) {
        const std::size_t         thisChunk = std::min(chunk, remaining);
        std::vector<std::uint8_t> values(thisChunk);

        in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size()));
        const std::streamsize nRead = in.gcount();
        if (nRead <= 0) {
            break;
        }
        const std::size_t nReadSize = static_cast<std::size_t>(nRead);
        if (nReadSize < values.size()) {
            values.resize(nReadSize);
        }
        pushData(state.get(), std::move(values));
        if (in.eof()) {
            break;
        }
        if (nReadSize < remaining) {
            remaining -= nReadSize;
        } else {
            break;
        }
    }
    if (in.bad()) {
        in.close();
        pushErrorFinal(state.get(), std::format("I/O error while reading file: {}", state->uri));
    } else {
        in.close();
        pushFinal(state.get());
    }
}

#if !defined(__EMSCRIPTEN__) && GR_HTTP_ENABLED
[[nodiscard]] inline bool runHttpGetNativeOnce(std::shared_ptr<ReaderState> state) {
    if (state == nullptr) {
        return false;
    }

    bool         dataReceived = false;
    cpr::Session session;
    session.SetUrl(cpr::Url{state->uri});
    session.SetTimeout(cpr::Timeout{static_cast<std::int32_t>(state->config.httpTimeoutNanos / 1'000'000ull)});
    if (!state->config.httpHeaders.empty()) {
        cpr::Header header;
        for (const auto& [k, v] : state->config.httpHeaders) {
            header.emplace(k, v);
        }
        session.SetHeader(std::move(header));
    }
    session.SetVerifySsl(cpr::VerifySsl{state->config.tlsVerifyPeer});
    session.SetWriteCallback(cpr::WriteCallback{[state, &dataReceived](std::string_view chunk, intptr_t) -> bool {
        if (state->cancelRequested.load(std::memory_order_acquire)) {
            return false;
        }
        if (!chunk.empty()) {
            dataReceived = true;
            pushData(state.get(), std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(chunk.data()), chunk.size()));
        }
        return true;
    }});

    auto response = session.Get();

    if (state->cancelRequested.load(std::memory_order_acquire)) {
        return false;
    }

    if (response.status_code >= 400) {
        pushError(state.get(), std::format("HTTP {} (body size: {})", response.status_code, response.text.size()));
        return false;
    }

    // dataReceived check is a workaround, cpr always ends with r.error.code == 1000
    if (!dataReceived && response.error.code != cpr::ErrorCode::OK) {
        pushError(state.get(), std::format("cpr error code:{}, cpr error message:{}, response status_code:{}", response.error.code, response.error.message, response.status_code));
        return false;
    }

    return true;
}

inline void runHttpGetNative(std::shared_ptr<ReaderState> state) {
    if (state == nullptr) {
        return;
    }

    if (!state->config.longPolling) {
        std::ignore = runHttpGetNativeOnce(state);
        pushFinal(state.get());
    } else { // long polling
        while (!state->cancelRequested.load(std::memory_order_acquire)) {
            const bool ok = runHttpGetNativeOnce(state);
            if (!ok) {
                break; // cancel or error. TODO send request on error or break?
            }
        }
        pushFinal(state.get());
    }
}

#endif

#if __EMSCRIPTEN__

inline void runHttpGetEmscriptenImpl(ReaderState* state) {
    if (state == nullptr) {
        return;
    }

    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    std::strcpy(attr.requestMethod, "GET");
    attr.attributes   = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.userData     = state;
    attr.timeoutMSecs = 0; // non-zero currently causes immediate onerror

    std::vector<const char*> headers;
    if (!state->config.httpHeaders.empty()) {
        headers.reserve(state->config.httpHeaders.size() * 2 + 1);
        for (const auto& [k, v] : state->config.httpHeaders) {
            headers.push_back(k.c_str());
            headers.push_back(v.c_str());
        }
        headers.push_back(nullptr);
        attr.requestHeaders = headers.data();
    } else {
        attr.requestHeaders = nullptr;
    }

    attr.onsuccess = [](emscripten_fetch_t* fetch) {
        auto* st = static_cast<ReaderState*>(fetch->userData);
        if (st == nullptr) {
            emscripten_fetch_close(fetch);
            return;
        }

        const bool longPolling = st->config.longPolling;

        if (!st->cancelRequested.load(std::memory_order_acquire) && fetch->data && fetch->numBytes > 0) {
            pushData(st, std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(fetch->data), static_cast<std::size_t>(fetch->numBytes)));
        }

        if (!longPolling || st->cancelRequested.load(std::memory_order_acquire)) {
            pushFinal(st);
            emscripten_fetch_close(fetch);
        } else {
            emscripten_fetch_close(fetch);
            runHttpGetEmscriptenImpl(st);
        }
    };

    attr.onerror = [](emscripten_fetch_t* fetch) {
        auto* st = static_cast<ReaderState*>(fetch->userData);
        if (st == nullptr) {
            emscripten_fetch_close(fetch);
            return;
        }

        if (st->cancelRequested.load(std::memory_order_acquire)) {
            pushError(st, std::format("fetch error (status {}). Cancelled by a user", fetch ? fetch->status : -1));
        } else {
            pushError(st, std::format("fetch error (status {})", fetch ? fetch->status : -1));
        }
        pushFinal(st);
        emscripten_fetch_close(fetch);
    };

    // clang-format off
    int hasFetch = EM_ASM_INT({ return (typeof globalThis.fetch === 'function') ? 1 : 0; });
    // clang-format on
    if (hasFetch == 0) {
        pushErrorFinal(state, "fetch is not available on this JS context");
        return;
    }
    emscripten_fetch_t* tmpFetch = emscripten_fetch(&attr, state->uri.c_str());
    if (tmpFetch == nullptr) {
        pushErrorFinal(state, "emscripten_fetch returned null emscripten_fetch_t");
        return;
    }
}

template<bool runOnMainThread = true>
void runHttpGetEmscripten(std::shared_ptr<ReaderState> state) {
    if (state == nullptr) {
        return;
    }
    if constexpr (runOnMainThread) {
        if (emscripten_is_main_runtime_thread()) {
            runHttpGetEmscriptenImpl(state.get());
        } else {
            emscripten_async_run_in_main_runtime_thread(
                EM_FUNC_SIG_IP,
                +[](void* st) {
                    runHttpGetEmscriptenImpl(static_cast<ReaderState*>(st));
                    return 0;
                },
                state.get());
        }
    } else {
        gr::thread_pool::Manager::defaultIoPool()->execute([state]() mutable { runHttpGetEmscriptenImpl(state.get()); });
    }
}

#endif

// Note for Emscripten: The returned `Reader` must remain alive until the underlying request has completed (final message received)
// or has been cancelled and fully processed.
[[nodiscard]] inline std::expected<Reader, gr::Error> readAsync(std::string_view uri, ReaderConfig config = {}) {
    if (detail::isDialogUri(uri)) {
        auto& dialogCallback = detail::dialogOpenCallback();
        if (!dialogCallback) {
            return std::unexpected(gr::Error{"dialog:/open used but no DialogOpenCallback registered"});
        }

        auto state = std::make_shared<ReaderState>(std::string(uri), config);

        state->dialogHandle      = std::make_unique<DialogOpenHandle>();
        DialogOpenHandle& handle = *state->dialogHandle;

        handle.completeWithMemory = [state](std::span<const std::uint8_t> bytes) { runReadMemorySource(state, bytes); };
        handle.completeWithFile   = [state](std::string path) {
            if (state != nullptr) {
                state->uri = std::move(path);
                gr::thread_pool::Manager::defaultIoPool()->execute([state]() mutable { runReadLocalFile(state); });
            }
        };
        handle.fail = [state](std::string_view msg) { pushErrorFinal(state.get(), msg); };

        dialogCallback(handle);
        return Reader{state};
    }

#if __EMSCRIPTEN__
    if (detail::isFileUri(uri)) {
        const auto newPath = detail::stripFileUri(uri);
        if (!newPath.has_value()) {
            return std::unexpected(newPath.error());
        }
        auto state = std::make_shared<ReaderState>(newPath.value(), config);
        gr::thread_pool::Manager::defaultIoPool()->execute([state]() mutable { runReadLocalFile(state); });
        return Reader{state};
    } else if (detail::isHttpUri(uri)) {
        auto state = std::make_shared<ReaderState>(std::string(uri), config);
        runHttpGetEmscripten(state);
        return Reader{state};
    } else {
        return std::unexpected(gr::Error{std::format("Something wrong with URI: {}", uri)});
    }
#else
    if (detail::isFileUri(uri)) {
        const auto newPath = detail::stripFileUri(uri);
        if (!newPath.has_value()) {
            return std::unexpected(newPath.error());
        }
        auto state = std::make_shared<ReaderState>(newPath.value(), config);
        gr::thread_pool::Manager::defaultIoPool()->execute([state]() mutable { runReadLocalFile(state); });
        return Reader{state};
    } else if (detail::isHttpUri(uri)) {
#if GR_HTTP_ENABLED
        auto state = std::make_shared<ReaderState>(std::string(uri), config);
        gr::thread_pool::Manager::defaultIoPool()->execute([state]() mutable { runHttpGetNative(state); });
        return Reader{state};
#else
        return std::unexpected(gr::Error{"HTTP(S) disabled at build time. See GR_HTTP_ENABLED for details."});
#endif
    } else {
        return std::unexpected(gr::Error{std::format("Something wrong with URI: {}", uri)});
    }
#endif
}

// Note: For in-memory Readers, all data and the final message are pushed synchronously inside readAsync(). There is no ongoing background
// work after readAsync() returns; cancel() has no effect in this case.
[[nodiscard]] inline std::expected<Reader, gr::Error> readAsync(std::span<const std::uint8_t> data, ReaderConfig config = {}, std::string_view logicalUri = "<memory>") {
    auto state = std::make_shared<ReaderState>(std::string(logicalUri), config);
    runReadMemorySource(state, data); // write all data directly to CircularBuffer, no extra copy
    return Reader{state};
}

[[nodiscard]] inline std::expected<Reader, gr::Error> readAsync(const std::uint8_t* ptr, std::size_t len, ReaderConfig config = {}, std::string_view logicalUri = "<memory>") { //
    if (ptr == nullptr) {
        return std::unexpected(gr::Error{"Failed to readAsync, data ptr == nullptr"});
    }

    return readAsync(std::span<const std::uint8_t>(ptr, len), std::move(config), logicalUri);
}

// Write to file and http POST

enum class WriteMode { overwrite, append };

struct WriterConfig {
    WriteMode                          mode             = WriteMode::overwrite;
    std::uint64_t                      httpTimeoutNanos = 30'000'000'000; // 30 s time-out for http(s)
    std::map<std::string, std::string> httpHeaders      = {};
    bool                               tlsVerifyPeer    = true; // for native https
};

struct WriteResult {
    // For local files, responseBody stays empty and httpStatus stays 0.
    std::string httpResponseBody; // HTTP response body (if any); empty for file writes
    long        httpStatus{0};    // HTTP status for POST; 0 for local file writes
};

struct WriterState {
    std::string               uri;
    WriterConfig              config;
    std::vector<std::uint8_t> data;

    std::atomic<bool>                     done{false};
    std::atomic<bool>                     cancelRequested{false};
    std::atomic<std::size_t>              updateCounter{0};
    std::expected<WriteResult, gr::Error> result; // Result is available when done == true

    WriterState(std::string uri_, WriterConfig config_) : uri(std::move(uri_)), config(config_) {}

#ifndef NDEBUG
    ~WriterState() {
        if (!done.load(std::memory_order_acquire)) {
            std::puts("ReaderState destroyed without final message");
        }
    }
#endif

    void setResult(std::expected<WriteResult, gr::Error> r) {
        result = std::move(r);
        done.store(true, std::memory_order_release);
        updateCounter.fetch_add(1, std::memory_order_release);
        updateCounter.notify_all();
    }
};

struct Writer {
    std::shared_ptr<WriterState> _state;

    Writer() = default;
    explicit Writer(std::shared_ptr<WriterState> state) : _state(std::move(state)) {}

    [[nodiscard]] bool finished() const { return _state == nullptr || _state->done.load(std::memory_order_acquire); }

    void cancel() {
        if (_state == nullptr) {
            return;
        }
        _state->cancelRequested.store(true, std::memory_order_release);
        _state->updateCounter.fetch_add(1, std::memory_order_release);
        _state->updateCounter.notify_all();
    }

    void wait() const {
        if (_state == nullptr) {
            return;
        }

#if __EMSCRIPTEN__
        // Do NOT block main WASM thread.
        if (emscripten_is_main_runtime_thread()) {
            std::println("fileio::Writer::wait() on main WASM thread is not allowed; returning immediately");
            return;
        }
#endif

        while (true) {
            if (_state->done.load(std::memory_order_acquire)) {
                return;
            }
            auto expected = _state->updateCounter.load(std::memory_order_acquire);
            _state->updateCounter.wait(expected, std::memory_order_acquire);
        }
    }

    // Non-blocking: if not finished, returns unexpected(...)
    [[nodiscard]] std::expected<WriteResult, gr::Error> result() const {
        if (_state == nullptr) {
            return std::unexpected(gr::Error{"Writer: no state"});
        }
        if (!_state->done.load(std::memory_order_acquire)) {
            return std::unexpected(gr::Error{"Writer: resultAsync() called before completion"});
        }
        return _state->result;
    }
};

#if !defined(__EMSCRIPTEN__) && GR_HTTP_ENABLED
[[nodiscard]] inline std::expected<WriteResult, gr::Error> runHttpPostNative(std::shared_ptr<WriterState> state) {
    if (state == nullptr) {
        return std::unexpected(gr::Error{"runHttpPostNative: state is nullptr"});
    }

    if (!detail::isHttpUri(state->uri)) {
        return std::unexpected(gr::Error{std::format("runHttpPostNative: URI is not HTTP(S): {}", state->uri)});
    }

    cpr::Session session;
    session.SetUrl(cpr::Url{state->uri});
    session.SetTimeout(cpr::Timeout{static_cast<std::int32_t>(state->config.httpTimeoutNanos / 1'000'000ull)});
    if (!state->config.httpHeaders.empty()) {
        cpr::Header header;
        for (const auto& [k, v] : state->config.httpHeaders) {
            header.emplace(k, v);
        }
        session.SetHeader(std::move(header));
    }
    session.SetVerifySsl(cpr::VerifySsl{state->config.tlsVerifyPeer});
    session.SetOption(cpr::BodyView{reinterpret_cast<const char*>(state->data.data()), state->data.size()});
    session.SetProgressCallback(cpr::ProgressCallback{[state](auto /*dlTotal*/, auto /*dlNow*/, auto /*ulTotal*/, auto /*ulNow*/, auto /*userData*/) { //
        return !state->cancelRequested.load(std::memory_order_acquire);
    }});

    auto response = session.Post();
    if (state->cancelRequested.load(std::memory_order_acquire)) {
        return std::unexpected(gr::Error{"runHttpPostNative: cancelled by user"});
    }

    if (response.status_code >= 400) {
        return std::unexpected(gr::Error{std::format("runHttpPostNative: HTTP {} (response size: {})", response.status_code, response.text.size())});
    }

    // response.text.empty() check is a workaround, cpr always ends with r.error.code == 1000
    if (response.text.empty() && response.error.code != cpr::ErrorCode::OK) {
        return std::unexpected(gr::Error{std::format("runHttpPostNative: cpr POST error code:{}, message:{}, http_status:{}", response.error.code, response.error.message, response.status_code)});
    }

    WriteResult res;
    res.httpStatus       = response.status_code;
    res.httpResponseBody = std::move(response.text);
    return res;
}
#endif

[[nodiscard]] inline std::expected<WriteResult, gr::Error> runWriteLocalFile(std::shared_ptr<WriterState> state) {
    namespace fs = std::filesystem;

    const fs::path  path{state->uri};
    const fs::path  parent = path.parent_path();
    std::error_code ec;
    if (!parent.empty()) {
        fs::create_directories(parent, ec);
        if (ec) {
            return std::unexpected(gr::Error{std::format("runWriteLocalFile: failed to create directory '{}': ec={}", parent.string(), ec.value())});
        }
    }

    if (state->config.mode == WriteMode::append) {
        if (fs::exists(path, ec) && !ec) {
            if (!fs::is_regular_file(path, ec) || ec) {
                return std::unexpected(gr::Error{std::format("runWriteLocalFile: path is not a regular file: {}", path.string())});
            }
        }
    }

    std::ios::openmode flags = std::ios::binary;
    switch (state->config.mode) {
    case WriteMode::overwrite: flags |= std::ios::trunc; break;
    case WriteMode::append: flags |= std::ios::app; break;
    }

    std::ofstream out(path, flags);
    if (!out) {
        return std::unexpected(gr::Error{std::format("runWriteLocalFile: failed to open file for writing: {}", path.string())});
    }

    if (!state->data.empty()) {
        out.write(reinterpret_cast<const char*>(state->data.data()), static_cast<std::streamsize>(state->data.size()));
        if (!out) {
            return std::unexpected(gr::Error{std::format("runWriteLocalFile: failed to write {} bytes to file: {}", state->data.size(), path.string())});
        }
    }

    out.close();
    return WriteResult{}; // For local files, responseBody stays empty and httpStatus stays 0.
}

#if __EMSCRIPTEN__
namespace detail {

inline void runHttpPostEmscriptenImpl(WriterState* state) {
    if (state == nullptr) {
        return;
    }

    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    std::strcpy(attr.requestMethod, "POST");
    attr.attributes   = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.timeoutMSecs = 0; // non-zero currently causes immediate onerror in our tests
    attr.userData     = state;

    std::vector<const char*> headers;
    if (!state->config.httpHeaders.empty()) {
        headers.reserve(state->config.httpHeaders.size() * 2 + 1);
        for (const auto& [k, v] : state->config.httpHeaders) {
            headers.push_back(k.c_str());
            headers.push_back(v.c_str());
        }
        headers.push_back(nullptr);
        attr.requestHeaders = headers.data();
    } else {
        attr.requestHeaders = nullptr;
    }

    attr.requestData     = state->data.empty() ? nullptr : reinterpret_cast<char*>(state->data.data());
    attr.requestDataSize = state->data.size();

    attr.onsuccess = [](emscripten_fetch_t* fetch) {
        auto* st = static_cast<WriterState*>(fetch->userData);
        if (st == nullptr) {
            emscripten_fetch_close(fetch);
            return;
        }
        int status = fetch != nullptr ? fetch->status : -1;

        std::expected<WriteResult, gr::Error> r;
        if (status < 200 || status >= 400) {
            r = std::unexpected(gr::Error{std::format("httpUploadPost (WASM): HTTP POST failed with status {}", status)});
        } else {
            WriteResult res;
            res.httpStatus = status;
            if (fetch && fetch->data && fetch->numBytes > 0) {
                res.httpResponseBody.assign(fetch->data, fetch->data + fetch->numBytes);
            }
            r = std::move(res);
        }

        if (st->cancelRequested.load(std::memory_order_acquire)) {
            r = std::unexpected(gr::Error{"writeAsync: cancelled by user"});
        }

        st->setResult(std::move(r));
        emscripten_fetch_close(fetch);
    };

    attr.onerror = [](emscripten_fetch_t* fetch) {
        auto* st = static_cast<WriterState*>(fetch->userData);
        if (st == nullptr) {
            emscripten_fetch_close(fetch);
            return;
        }

        int  status = fetch != nullptr ? fetch->status : -1;
        auto r      = std::unexpected(gr::Error{std::format("httpUploadPost (WASM): fetch error, status {}", status)});

        if (st->cancelRequested.load(std::memory_order_acquire)) {
            r = std::unexpected(gr::Error{"writeAsync: cancelled by user"});
        }

        st->setResult(std::move(r));
        emscripten_fetch_close(fetch);
    };

    // clang-format off
    int hasFetch = EM_ASM_INT({ return (typeof globalThis.fetch === 'function') ? 1 : 0; });
    // clang-format on
    if (!hasFetch) {
        auto r = std::unexpected(gr::Error{"httpUploadPost (WASM): fetch is not available in this JS context"});
        state->setResult(std::move(r));
        return;
    }

    emscripten_fetch_t* tmpFetch = emscripten_fetch(&attr, state->uri.c_str());
    if (!tmpFetch) {
        auto r = std::unexpected(gr::Error{"httpUploadPost (WASM): emscripten_fetch returned null emscripten_fetch_t"});
        state->setResult(std::move(r));
    }
}

template<bool runOnMainThread = true>
inline void runHttpPostEmscripten(std::shared_ptr<WriterState> state) {
    if (!state) {
        return;
    }

    if constexpr (runOnMainThread) {
        if (emscripten_is_main_runtime_thread()) {
            runHttpPostEmscriptenImpl(state.get());
        } else {
            emscripten_async_run_in_main_runtime_thread(
                EM_FUNC_SIG_IP,
                +[](void* st) {
                    runHttpPostEmscriptenImpl(static_cast<WriterState*>(st));
                    return 0;
                },
                state.get());
        }
    } else {
        gr::thread_pool::Manager::defaultIoPool()->execute([state]() mutable { runHttpPostEmscriptenImpl(state.get()); });
    }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdollar-in-identifier-extension"
// Browser "download to disk"
inline void runDownloadEmscriptenImpl(WriterState* st) {
    if (st == nullptr) {
        return;
    }

    const auto len = static_cast<int>(st->data.size());
    if (len <= 0) {
        st->setResult(WriteResult{});
        return;
    }

    // clang-format off
    EM_ASM(
        {
            const filename = UTF8ToString($0);
            const ptr      = $1 >>> 0;
            const len      = $2 >>> 0;

            const array = new Uint8Array(len);
            array.set(HEAPU8.subarray(ptr, ptr + len));

            const blob = new Blob([array], {
                type:
                    'application/octet-stream'
            });
            const link    = document.createElement('a');
            link.href     = URL.createObjectURL(blob);
            link.download = filename;
            document.body.appendChild(link);

            try {
                link.click();
            } catch (e) {
                console.error("gr::fileio: download link.click() failed:", e);
            }

            try {
                document.body.removeChild(link);
            } catch (e) {
                console.error("gr::fileio: download removeChild failed:", e);
            }

            try {
                URL.revokeObjectURL(link.href);
            } catch (e) {
                console.error("gr::fileio: download revokeObjectURL failed:", e);
            }
        },
        st->uri.c_str(), st->data.data(), len);
    // clang-format on

    st->setResult(WriteResult{});
}
#pragma GCC diagnostic pop

inline void runDownloadEmscripten(std::shared_ptr<WriterState> state) {
    if (state == nullptr) {
        return;
    }

    if (isMainThread()) {
        runDownloadEmscriptenImpl(state.get());
    } else {
        emscripten_async_run_in_main_runtime_thread(EM_FUNC_SIG_VI, +[](void* p) { runDownloadEmscriptenImpl(static_cast<WriterState*>(p)); }, state.get());
    }
}

} // namespace detail
#endif // __EMSCRIPTEN__

[[nodiscard]] inline std::expected<Writer, gr::Error> writeAsync(std::string_view uri, std::span<const std::uint8_t> data, const WriterConfig& config = {}) {

#if __EMSCRIPTEN__
    if (detail::isHttpUri(uri)) {
        if (config.mode != WriteMode::overwrite) {
            return std::unexpected(gr::Error{"append mode is not supported for HTTP(S) URIs"});
        }
        auto state = std::make_shared<WriterState>(std::string(uri), config);
        state->data.assign(data.begin(), data.end());
        detail::runHttpPostEmscripten(state);
        return Writer{state};
    } else if (detail::isDownloadUri(uri)) {
        const auto filename = detail::stripDownloadUri(uri);
        if (!filename.has_value()) {
            return std::unexpected(filename.error());
        }

        auto state = std::make_shared<WriterState>(filename.value(), config);
        state->data.assign(data.begin(), data.end());

        detail::runDownloadEmscripten(state);
        return Writer{state};

    } else if (detail::isFileUri(uri)) {
        const auto newPath = detail::stripFileUri(uri);
        if (!newPath.has_value()) {
            return std::unexpected(newPath.error());
        }
        auto state = std::make_shared<WriterState>(newPath.value(), config);
        state->data.assign(data.begin(), data.end());
        gr::thread_pool::Manager::defaultIoPool()->execute([state]() mutable {
            auto r = runWriteLocalFile(state);
            if (state->cancelRequested.load(std::memory_order_acquire)) {
                r = std::unexpected(gr::Error{"cancelled by user"});
            }
            state->setResult(std::move(r));
        });
        return Writer{state};
    } else {
        return std::unexpected(gr::Error{std::format("Something wrong with URI: {}", uri)});
    }
#else

    if (detail::isHttpUri(uri)) {
#if GR_HTTP_ENABLED
        if (config.mode != WriteMode::overwrite) {
            return std::unexpected(gr::Error{"append mode is not supported for HTTP(S) URIs"});
        }
        auto state = std::make_shared<WriterState>(std::string(uri), config);
        state->data.assign(data.begin(), data.end());
        gr::thread_pool::Manager::defaultIoPool()->execute([state]() mutable {
            auto r = runHttpPostNative(state);
            state->setResult(std::move(r));
        });
        return Writer{state};
#else
        return std::unexpected(gr::Error{"HTTP(S) disabled at build time. See GR_HTTP_ENABLED for details."});
#endif
    } else if (detail::isDownloadUri(uri)) {
        const auto filenameExp = detail::stripDownloadUri(uri);
        if (!filenameExp.has_value()) {
            return std::unexpected(filenameExp.error());
        }

        const auto path  = detail::resolveDownloadPath(filenameExp.value());
        auto       state = std::make_shared<WriterState>(path, config);
        state->data.assign(data.begin(), data.end());

        gr::thread_pool::Manager::defaultIoPool()->execute([state]() mutable {
            auto r = runWriteLocalFile(state);
            if (state->cancelRequested.load(std::memory_order_acquire)) {
                r = std::unexpected(gr::Error{"cancelled by user"});
            }
            state->setResult(std::move(r));
        });
        return Writer{state};

    } else if (detail::isFileUri(uri)) {
        const auto newPath = detail::stripFileUri(uri);
        if (!newPath.has_value()) {
            return std::unexpected(newPath.error());
        }
        auto state = std::make_shared<WriterState>(newPath.value(), config);
        state->data.assign(data.begin(), data.end());
        gr::thread_pool::Manager::defaultIoPool()->execute([state]() mutable {
            auto r = runWriteLocalFile(state);
            if (state->cancelRequested.load(std::memory_order_acquire)) {
                r = std::unexpected(gr::Error{"cancelled by user"});
            }
            state->setResult(std::move(r));
        });
        return Writer{state};
    } else {
        return std::unexpected(gr::Error{std::format("Something wrong with URI: {}", uri)});
    }
#endif
}

[[nodiscard]] inline std::expected<WriteResult, gr::Error> write(std::string_view uri, std::span<const std::uint8_t> data, const WriterConfig& config = {}) {

#if __EMSCRIPTEN__
    // Do NOT block main WASM thread.
    if (emscripten_is_main_runtime_thread()) {
        return std::unexpected(gr::Error{"fileio::write() can not be called from main thread, it is not allowed to block main WASM thread"});
    }
#endif

    auto writerExp = writeAsync(uri, data, config);
    if (!writerExp.has_value()) {
        return std::unexpected(writerExp.error());
    }
    auto writer = std::move(writerExp.value());
    writer.wait();
    return writer.result();
}

} // namespace gr::algorithm::fileio
#endif // FILEIO_HPP
