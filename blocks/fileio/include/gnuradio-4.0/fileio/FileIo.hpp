#ifndef FILEIO_HPP
#define FILEIO_HPP

#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/fileio/FileIoHelpers.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

#include <algorithm>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#define GR4_FILEIO_CPR_AVAILABLE 1

#if GR4_FILEIO_CPR_AVAILABLE
#include <cpr/cpr.h>
#endif
#if __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#include <emscripten/fetch.h>
#include <emscripten/html5.h>
#endif

namespace gr::blocks::fileio {

struct RequestOptions {
    std::size_t                  chunkBytes    = 64 * 1024;
    std::optional<std::uint64_t> offset        = std::nullopt;
    int                          timeoutMs     = 30000; // native HTTP timeout
    std::size_t                  bufferMinSize = 1024;  // one gr::Message per slot
};

struct SubscriptionState {
    std::atomic<bool> running{true};
    std::string       uri;
    RequestOptions    opts;

    gr::CircularBuffer<gr::Message> buffer;
    decltype(buffer.new_writer())   writer;
    decltype(buffer.new_reader())   reader;

#if !GR4_FILEIO_WASM
    std::thread worker;
#else
    emscripten_fetch_t* fetchHandle{nullptr};
#endif

    // reader status, we assume single-consumer
    std::optional<gr::Error> lastError;
    bool                     sawFinal = false;

    SubscriptionState(std::string uri_, RequestOptions opt) : uri(std::move(uri_)), opts(opt), buffer(opt.bufferMinSize), writer(buffer.new_writer()), reader(buffer.new_reader()) {}
};

struct Subscription {

private:
    std::shared_ptr<SubscriptionState> _state;
    static constexpr std::string_view  kDataKey = "data";

public:
    Subscription() = default;
    explicit Subscription(std::shared_ptr<SubscriptionState> state) : _state(std::move(state)) {}
    Subscription(Subscription&&) noexcept            = default;
    Subscription& operator=(Subscription&&) noexcept = default;
    ~Subscription() { stopAndJoin(); }

    bool                             isRunning() const { return _state && _state->running.load(std::memory_order_acquire); }
    gr::CircularBuffer<gr::Message>& buffer() const { return _state->buffer; }

    void cancel() {
        if (!_state) {
            return;
        }
        _state->running.store(false, std::memory_order_relaxed);
#if GR4_FILEIO_WASM
        if (_state->fetchHandle) {
            emscripten_fetch_close(_state->fetchHandle);
            _state->fetchHandle = nullptr;
        }
#endif
    }

    bool                            finished() const { return !_state || _state->sawFinal || !_state->running.load(std::memory_order_acquire); }
    const std::optional<gr::Error>& error() const { return _state->lastError; }

    std::optional<std::vector<std::uint8_t>> poll() {
        if (!_state) {
            return std::nullopt;
        }

        auto span = _state->reader.get();
        if (span.empty()) {
            return std::nullopt;
        }

        std::vector<std::uint8_t> out;
        for (const auto& msg : span) {
            if (!msg.data.has_value()) {
                _state->lastError = msg.data.error();
                continue;
            }
            auto it = msg.data->find(std::string(kDataKey));
            if (it == msg.data->end()) {
                continue;
            }
            const std::vector<std::uint8_t>* bytes = std::get_if<std::vector<std::uint8_t>>(&it->second);
            if (bytes != nullptr) {
                out.insert(out.end(), bytes->begin(), bytes->end());
            }
            if (msg.cmd == gr::message::Command::Final) {
                _state->sawFinal = true;
            }
        }
        std::ignore = span.consume(span.size());
        if (_state->lastError) {
            return std::nullopt;
        }
        return out;
    }

private:
    void stopAndJoin() {
        if (!_state) {
            return;
        }
        _state->running.store(false, std::memory_order_relaxed);
#if GR4_FILEIO_WASM
        if (_state->fetchHandle) {
            emscripten_fetch_close(_state->fetchHandle);
            _state->fetchHandle = nullptr;
        }
#else
        if (_state->worker.joinable()) {
            _state->worker.join();
        }
#endif
    }
};

inline void publishMessage(SubscriptionState* st, gr::Message&& m) {
#if GR4_FILEIO_WASM
    // Do not block browser thread; best-effort
    auto span = st->writer.template tryReserve<gr::SpanReleasePolicy::ProcessNone>(1);
    if (span.size() == 1) {
        span[0] = std::move(m);
        span.publish(1);
    }
#else
    auto span = st->writer.template reserve<gr::SpanReleasePolicy::ProcessNone>(1);
    span[0]   = std::move(m);
    span.publish(1);
#endif
}

inline void pushData(SubscriptionState* state, const std::uint8_t* pData, std::size_t n) {
    if (!state || !state->running.load(std::memory_order_relaxed) || n == 0) {
        return;
    }

    gr::Message msg;
    msg.endpoint = state->uri;
    msg.cmd      = gr::message::Command::Notify;

    std::vector<std::uint8_t> values(n);
    std::memcpy(values.data(), pData, n);
    msg.data = gr::property_map{{"data", std::move(values)}};
    publishMessage(state, std::move(msg));
}

inline void pushError(SubscriptionState* state, std::string_view msg) {
    if (!state) {
        return;
    }
    gr::Message errorMsg;
    errorMsg.endpoint = state->uri;
    errorMsg.cmd      = gr::message::Command::Notify;
    errorMsg.data     = std::unexpected(gr::Error{std::move(msg)});
    publishMessage(state, std::move(errorMsg));
}

inline void pushFinal(SubscriptionState* state) {
    gr::Message finalMsg;
    finalMsg.endpoint = state->uri;
    finalMsg.cmd      = gr::message::Command::Final;
    finalMsg.data     = gr::property_map{};
    publishMessage(state, std::move(finalMsg));
}

inline void pushErrorFinal(SubscriptionState* state, std::string_view msg) {
    pushError(state, msg);
    pushFinal(state);
}

#if !GR4_FILEIO_WASM
inline void runNativeLocalFile(std::shared_ptr<SubscriptionState> state) {
    namespace fs = std::filesystem;

    fs::path path = state->uri;

    std::error_code ec;
    if (!fs::exists(path, ec) || !fs::is_regular_file(path, ec)) {
        pushErrorFinal(state.get(), std::format("file not found or not a regular file: {}", state->uri));
        state->running = false;
        return;
    }

    const auto fileSize = fs::file_size(path, ec);
    if (ec) {
        pushErrorFinal(state.get(), std::format("failed to get file size: {} (ec={})", state->uri, ec.value()));
        state->running = false;
        return;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        pushErrorFinal(state.get(), std::format("failed to open file: {}", state->uri));
        state->running = false;
        return;
    }

    std::uint64_t start = state->opts.offset.value_or(0);
    if (start >= fileSize) {
        pushFinal(state.get());
        state->running = false;
        return;
    }
    if (start > 0) {
        in.seekg(static_cast<std::streamoff>(start), std::ios::beg);
        if (!in) {
            pushErrorFinal(state.get(), std::format("seek failed to offset {} in file: {}", start, state->uri));
            state->running = false;
            return;
        }
    }

    std::vector<std::uint8_t> buf(std::max<std::size_t>(1, state->opts.chunkBytes));
    while (state->running.load(std::memory_order_relaxed)) {
        in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        const std::streamsize n = in.gcount();
        if (n <= 0) {
            break;
        }
        pushData(state.get(), buf.data(), static_cast<std::size_t>(n));
        if (in.eof()) {
            break;
        }
    }

    in.close();
    pushFinal(state.get());
    state->running = false;
}

#if GR4_FILEIO_CPR_AVAILABLE
inline void runNativeHttp(std::shared_ptr<SubscriptionState> state) {
    cpr::Session session;
    session.SetUrl(cpr::Url{state->uri});
    session.SetTimeout(cpr::Timeout{state->opts.timeoutMs});

    if (state->opts.offset && *state->opts.offset > 0) {
        session.SetHeader({{"Range", "bytes=" + std::to_string(*state->opts.offset) + "-"}});
    }

    std::atomic<bool> ok{true};

    session.SetWriteCallback(cpr::WriteCallback{[state, &ok](const std::string_view& chunk, long) -> bool {
        std::println("WriteCallback chunk:{}", chunk);
        if (!state->running.load(std::memory_order_relaxed)) {
            ok = false;
            return false;
        }
        if (!chunk.empty()) {
            pushData(state.get(), reinterpret_cast<const std::uint8_t*>(chunk.data()), chunk.size());
        }
        return true;
    }});

    auto r = session.Get();

    if (!state->running.load(std::memory_order_relaxed)) {
        return;
    }

    if (r.error.code != cpr::ErrorCode::OK || r.status_code >= 400 || !ok.load()) {
        const std::string msg = (r.error.code != cpr::ErrorCode::OK) ? r.error.message : ("HTTP " + std::to_string(r.status_code));
        pushErrorFinal(state.get(), msg);
        state->running = false;
        return;
    }

    pushFinal(state.get());
    state->running = false;
}
#endif // GR4_FILEIO_CPR_AVAILABLE
#endif // !WASM

#if GR4_FILEIO_WASM
inline void runWasmHttp(SubscriptionState* st) {
    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    std::strcpy(attr.requestMethod, "GET");
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.userData   = st;

    std::string rangeHeader;
    const char* hdrs[2] = {nullptr, nullptr};
    if (st->opts.offset && *st->opts.offset > 0) {
        rangeHeader         = "Range: bytes=" + std::to_string(*st->opts.offset) + "-";
        hdrs[0]             = rangeHeader.c_str();
        attr.requestHeaders = hdrs;
    }

    attr.onsuccess = [](emscripten_fetch_t* fetch) {
        auto* st = static_cast<SubscriptionState*>(fetch->userData);
        if (!st->running.load(std::memory_order_relaxed)) {
            emscripten_fetch_close(fetch);
            return;
        }
        if (fetch->numBytes > 0 && fetch->data) {
            pushData(st, reinterpret_cast<const uint8_t*>(fetch->data), static_cast<std::size_t>(fetch->numBytes));
        }
        pushFinal(st);
        st->running = false;
        emscripten_fetch_close(fetch);
    };
    attr.onerror = [](emscripten_fetch_t* fetch) {
        auto* st = static_cast<SubscriptionState*>(fetch->userData);
        pushErrorFinal(st, std::format("fetch error, status {}", fetch->status));
        st->running = false;
        emscripten_fetch_close(fetch);
    };

    st->fetchHandle = emscripten_fetch(&attr, st->uri.c_str());
}
#endif

inline std::expected<Subscription, gr::Error> subscribe(std::string_view uri, RequestOptions opts = {}) {
    auto state = std::make_shared<SubscriptionState>(std::string(uri), opts);

#if GR4_FILEIO_WASM
    if (!detail::isHttpUri(state->uri)) {
        return std::unexpected(gr::Error{"WASM supports only http(s) sources"});
    }
    runWasmHttp(state.get());
    return Subscription{std::move(state)};
#else
    if (detail::isHttpUri(state->uri)) {
#if GR4_FILEIO_CPR_AVAILABLE
        state->worker = std::thread([state] { runNativeHttp(state); });
        return Subscription{std::move(state)};
#else
        return std::unexpected(gr::Error{"HTTP(S) disabled at build time"});
#endif
    } else {
        if (detail::isFileUri(state->uri)) {
            state->uri = detail::stripFileUri(state->uri);
        }
        state->worker = std::thread([state] { runNativeLocalFile(state); });
        return Subscription{std::move(state)};
    }
#endif
}

} // namespace gr::blocks::fileio
#endif // FILEIO_HPP
