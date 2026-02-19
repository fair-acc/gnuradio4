#ifndef GNURADIO_HTTP_BLOCK_HPP
#define GNURADIO_HTTP_BLOCK_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/BlockingEventSync.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <queue>
#include <semaphore>

#ifdef __GNUC__
#pragma GCC diagnostic push // ignore warning of external libraries that from this lib-context we do not have any control over
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

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#include <emscripten/fetch.h>
#include <emscripten/threading.h>
#else
#include <httplib.h>
#endif

using namespace gr;
using namespace std::chrono_literals;

namespace gr::http {

enum class RequestType : char {
    GET       = 1,
    SUBSCRIBE = 2,
    POST      = 3,
};

GR_REGISTER_BLOCK(gr::http::HttpBlock, [T], [ float, double ])

template<typename T>
struct HttpBlock : Block<HttpBlock<T>>, BlockingEventSync<HttpBlock<T>> {
    using Description = Doc<R""(
The HttpBlock allows to use the responses from HTTP APIs (e.g. REST APIs) as the value for this block's output port.
The block can be used either on-demand to do single requests, or can use long polling to subscribe to an event stream.
The result is provided on a single output port as a map with the following keys:
- status: The HTTP status code, usually 200 on success
- raw-data: The data of the response
- mime-type: The mime-type of the response
)"">;

    using Block<HttpBlock<T>>::Block; // needed to inherit mandatory base-class Block(property_map) constructor

    PortOut<pmt::Value::Map> out;

    std::string           url;
    std::string           endpoint = "/";
    gr::http::RequestType type     = gr::http::RequestType::GET;
    std::string           parameters; // x-www-form-urlencoded encoded POST parameters

    GR_MAKE_REFLECTABLE(HttpBlock, out, url, endpoint, type, parameters);

    // used for queuing GET responses for the consumer
    std::queue<pmt::Value::Map> _backlog;
    std::mutex                  _backlog_mutex;

    std::atomic_size_t    _pendingRequests = 0;
    std::binary_semaphore _ready{0};

#ifndef __EMSCRIPTEN__
    std::unique_ptr<httplib::Client> _client;
#endif

#ifdef __EMSCRIPTEN__
    void queueWorkEmscripten(emscripten_fetch_t* fetch) {
        pmt::Value::Map result;
        result["mime-type"] = "text/plain";
        result["status"]    = static_cast<int>(fetch->status);
        result["raw-data"]  = std::string(fetch->data, static_cast<std::size_t>(fetch->numBytes));

        queueWork(result);
    }

    void onSuccess(emscripten_fetch_t* fetch) {
        queueWorkEmscripten(fetch);
        _ready.release();
        emscripten_fetch_close(fetch);
    }

    void onError(emscripten_fetch_t* fetch) {
        // we still want to queue the response, the statusCode will just not be 200
        queueWorkEmscripten(fetch);
        _ready.release();
        emscripten_fetch_close(fetch);
    }

    void doRequestEmscripten() {
        emscripten_fetch_attr_t attr;
        emscripten_fetch_attr_init(&attr);
        if (type == RequestType::POST) {
            strcpy(attr.requestMethod, "POST");
            if (!parameters.empty()) {
                attr.requestData     = parameters.c_str();
                attr.requestDataSize = parameters.size();
            }
        } else {
            strcpy(attr.requestMethod, "GET");
        }

        // this is needed so that we can call into member functions again, when we receive the Fetch callback
        attr.userData = this;

        attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
        attr.onsuccess  = [](emscripten_fetch_t* fetch) {
            auto src = static_cast<HttpBlock<T>*>(fetch->userData);
            src->onSuccess(fetch);
        };
        attr.onerror = [](emscripten_fetch_t* fetch) {
            auto src = static_cast<HttpBlock<T>*>(fetch->userData);
            src->onError(fetch);
        };
        const auto target = url + endpoint;
        std::ignore       = emscripten_fetch(&attr, target.c_str());
    }

    void runThreadEmscripten() {
        if (type == RequestType::SUBSCRIBE) {
            while (!this->eventStopRequested()) {
                doRequestEmscripten();
                _ready.acquire();
            }
        } else {
            while (!this->eventStopRequested()) {
                _ready.acquire();
                std::size_t nRequests = _pendingRequests.exchange(0UZ, std::memory_order_acq_rel);
                while (nRequests-- > 0UZ) {
                    doRequestEmscripten();
                }
            }
        }
    }
#else
    void runThreadNative() {
        _client = std::make_unique<httplib::Client>(url);
        _client->set_follow_location(true);
        if (type == RequestType::SUBSCRIBE) {
            // it's long polling, be generous with timeouts
            _client->set_read_timeout(1h);
            _client->Get(endpoint, [&](const char* data, size_t len) {
                pmt::Value::Map result;
                result["mime-type"] = "text/plain";
                result["status"]    = 200;
                result["raw-data"]  = std::string(data, len);

                queueWork(result);

                return !this->eventStopRequested();
            });
        } else {
            while (!this->eventStopRequested()) {
                _ready.acquire();
                std::size_t nRequests = _pendingRequests.exchange(0UZ, std::memory_order_acq_rel);
                while (nRequests-- > 0UZ) {
                    httplib::Result resp;
                    if (type == RequestType::POST) {
                        resp = parameters.empty() ? _client->Post(endpoint) : _client->Post(endpoint, parameters, "application/x-www-form-urlencoded");
                    } else {
                        resp = _client->Get(endpoint);
                    }
                    pmt::Value::Map result;
                    if (resp) {
                        result["mime-type"] = "text/plain";
                        result["status"]    = resp->status;
                        result["raw-data"]  = resp->body;
                        queueWork(result);
                    }
                }
            }
        }
    }
#endif

    void queueWork(const pmt::Value::Map& item) {
        {
            std::lock_guard lg{_backlog_mutex};
            _backlog.push(item);
        }
        this->eventNotify();
    }

    void startThread() {
        const bool started = this->eventRunInIoPool(std::format("uT:{}", gr::meta::shorten_type_name(this->unique_name)), [this]() {
#ifdef __EMSCRIPTEN__
            runThreadEmscripten();
#else
            runThreadNative();
#endif
        });
        if (!started) {
            this->emitErrorMessage("startThread()", "HTTP worker already running");
        }
    }

    void stopThread() {
        this->eventSyncStop();
        _ready.release();
#ifndef __EMSCRIPTEN__
        if (_client) {
            _client->stop();
        }
#endif
    }

    void startWorker() {
        if (this->eventWorkerRunning()) {
            return;
        }
        this->eventSyncStart();
        startThread();
    }

    ~HttpBlock() { stopThread(); }

    void settingsChanged(const property_map& /*oldSettings*/, property_map& newSettings) {
        if (newSettings.contains("url") || newSettings.contains("type")) { // other setting changes are hot-swappable without restarting the Client
            if (this->eventWorkerRunning()) {
                stopThread();
                startWorker();
            }
        }
    }

    void start() { startWorker(); }

    void stop() { stopThread(); }

    [[nodiscard]] constexpr auto processOne() noexcept {
        pmt::Value::Map result;
        std::lock_guard lg{_backlog_mutex};
        if (!_backlog.empty()) {
            result = _backlog.front();
            _backlog.pop();
        }
        return result;
    }

    void trigger() {
        _pendingRequests.fetch_add(1UZ, std::memory_order_acq_rel);
        _ready.release();
    }

    void processMessages(gr::MsgPortInBuiltin& port, std::span<const gr::Message> message) {
        gr::Block<HttpBlock<T>>::processMessages(port, message);

        std::ranges::for_each(message, [this](auto& m) {
            if (type == RequestType::SUBSCRIBE) {
                if (m.data.has_value() && m.data.value().contains("active")) {
                    // for long polling, the subscription should stay active, if and only if the messages' "active" member is true
                    if (m.data.value().at("active").value_or(false)) {
                        startWorker();
                    } else {
                        stopThread();
                    }
                }
            } else {
                // for all other modes, an incoming message means to trigger a new request
                trigger();
            }
        });
    }
};

} // namespace gr::http

#endif // GNURADIO_HTTP_BLOCK_HPP
