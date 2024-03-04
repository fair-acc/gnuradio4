#ifndef GNURADIO_HTTP_BLOCK_HPP
#define GNURADIO_HTTP_BLOCK_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/reflection.hpp>
#include <pmtv/pmt.hpp>

#include <semaphore>
#include <queue>

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
    GET = 1,
    SUBSCRIBE = 2,
    POST = 3,
};

using HttpBlockDoc = Doc<R""(
The HttpBlock allows to use the responses from HTTP APIs (e.g. REST APIs) as the value for this block's output port.
The block can be used either on-demand to do single requests, or can use long polling to subscribe to an event stream.
The result is provided on a single output port as a map with the following keys:
- status: The HTTP status code, usually 200 on success
- raw-data: The data of the response
- mime-type: The mime-type of the response
)"">;

template<typename T>
class HttpBlock : public gr::Block<HttpBlock<T>, BlockingIO<false>, HttpBlockDoc> {
private:
    // used for queuing GET responses for the consumer
    std::queue<pmtv::map_t> _backlog;
    std::mutex              _backlog_mutex;

    std::shared_ptr<std::thread> _thread;
    std::atomic_size_t           _pendingRequests = 0;
    std::atomic_bool             _shutdownThread  = false;
    std::binary_semaphore        _ready{ 0 };

#ifndef __EMSCRIPTEN__
    std::unique_ptr<httplib::Client> _client;
#endif

#ifdef __EMSCRIPTEN__
    void
    queueWorkEmscripten(emscripten_fetch_t *fetch) {
        pmtv::map_t result;
        result["mime-type"] = "text/plain";
        result["status"]    = static_cast<int>(fetch->status);
        result["raw-data"]  = std::string(fetch->data, static_cast<std::size_t>(fetch->numBytes));

        queueWork(result);
    }

    void
    onSuccess(emscripten_fetch_t *fetch) {
        queueWorkEmscripten(fetch);
        emscripten_fetch_close(fetch);
    }

    void
    onError(emscripten_fetch_t *fetch) {
        // we still want to queue the response, the statusCode will just not be 200
        queueWorkEmscripten(fetch);
        emscripten_fetch_close(fetch);
    }

    void
    doRequestEmscripten() {
        emscripten_fetch_attr_t attr;
        emscripten_fetch_attr_init(&attr);
        if (_type == RequestType::POST) {
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
        attr.onsuccess  = [](emscripten_fetch_t *fetch) {
            auto src = static_cast<HttpBlock<T> *>(fetch->userData);
            src->onSuccess(fetch);
        };
        attr.onerror = [](emscripten_fetch_t *fetch) {
            auto src = static_cast<HttpBlock<T> *>(fetch->userData);
            src->onError(fetch);
        };
        const auto target = url + endpoint;
        std::ignore       = emscripten_fetch(&attr, target.c_str());
    }

    void
    runThreadEmscripten() {
        if (_type == RequestType::SUBSCRIBE) {
            while (!_shutdownThread) {
                // long polling, just keep doing requests
                std::thread thread{ &HttpBlock::doRequestEmscripten, this };
                thread.join();
            }
        } else {
            while (!_shutdownThread) {
                while (_pendingRequests > 0) {
                    _pendingRequests--;
                    std::thread thread{ &HttpBlock::doRequestEmscripten, this };
                    thread.join();
                }
                _ready.acquire();
            }
        }
    }
#else
    void
    runThreadNative() {
        _client = std::make_unique<httplib::Client>(url);
        _client->set_follow_location(true);
        if (_type == RequestType::SUBSCRIBE) {
            // it's long polling, be generous with timeouts
            _client->set_read_timeout(1h);
            _client->Get(endpoint, [&](const char *data, size_t len) {
                pmtv::map_t result;
                result["mime-type"] = "text/plain";
                result["status"]    = 200;
                result["raw-data"]  = std::string(data, len);

                queueWork(result);

                return !_shutdownThread;
            });
        } else {
            while (!_shutdownThread) {
                while (_pendingRequests > 0) {
                    _pendingRequests--;
                    httplib::Result resp;
                    if (_type == RequestType::POST) {
                        resp = parameters.empty() ? _client->Post(endpoint) : _client->Post(endpoint, parameters, "application/x-www-form-urlencoded");
                    } else {
                        resp = _client->Get(endpoint);
                    }
                    pmtv::map_t result;
                    if (resp) {
                        result["mime-type"] = "text/plain";
                        result["status"]    = resp->status;
                        result["raw-data"]  = resp->body;
                        queueWork(result);
                    }
                }

                _ready.acquire();
            }
        }
    }
#endif

    void
    queueWork(const pmtv::map_t &item) {
        {
            std::lock_guard lg{ _backlog_mutex };
            _backlog.push(item);
        }
        const auto work = this->invokeWork();
        if (work == work::Status::DONE) {
            this->requestStop();
        }
        this->ioLastWorkStatus.exchange(work, std::memory_order_relaxed);
    }

    void
    startThread() {
        if (_thread) {
            _thread.reset();
        }
        _thread = std::shared_ptr<std::thread>(new std::thread([this]() {
#ifdef __EMSCRIPTEN__
                                                   runThreadEmscripten();
#else
                                                   runThreadNative();
#endif
                                               }),
                                               [this](std::thread *t) {
                                                   if (auto ret = this->changeStateTo(gr::lifecycle::State::REQUESTED_STOP); !ret) {
                                                       throw std::invalid_argument(fmt::format("{}::startThread() could not change state to REQUESTED_STOP", this->name));
                                                   }
                                                   _shutdownThread = true;
                                                   _ready.release();
#ifndef __EMSCRIPTEN__
                                                   if (_client) {
                                                       _client->stop();
                                                   }
#endif
                                                   if (t->joinable()) {
                                                       t->join();
                                                   }
                                                   _shutdownThread = false;
                                                   delete t;
                                                    if (auto ret = this->changeStateTo(gr::lifecycle::State::STOPPED); !ret) {
                                                        throw std::invalid_argument(fmt::format("{}::startThread() could not change state to STOPPED", this->name));
                                                    }
                                               });
    }

    void
    stopThread() {
        _thread.reset();
    }

    gr::http::RequestType _type = gr::http::RequestType::GET;

public:
    PortOut<pmtv::map_t> out;

    std::string url;
    std::string endpoint   = "/";
    std::string type       = std::string(magic_enum::enum_name(_type));
    std::string parameters; // x-www-form-urlencoded encoded POST parameters

    ~HttpBlock() { stopThread(); }

    void
    settingsChanged(const property_map & /*oldSettings*/, property_map &newSettings) {
        if (newSettings.contains("url") || newSettings.contains("type")) {
            if (newSettings.contains("type")) {
                _type = magic_enum::enum_cast<gr::http::RequestType>(type, magic_enum::case_insensitive).value_or(_type);
            }
            // other setting changes are hot-swappable without restarting the Client
            startThread();
        }
    }

    void
    start() {
        startThread();
    }

    void
    stop() {
        stopThread();
    }

    [[nodiscard]] constexpr auto
    processOne() noexcept {
        pmtv::map_t     result;
        std::lock_guard lg{ _backlog_mutex };
        if (!_backlog.empty()) {
            result = _backlog.front();
            _backlog.pop();
        }
        return result;
    }

    void
    trigger() {
        _pendingRequests++;
        _ready.release();
    }

    void
    processMessages(gr::MsgPortInNamed<"__Builtin"> &, std::span<const gr::Message> message) {
        std::ranges::for_each(message, [this](auto &m) {
            if (_type == RequestType::SUBSCRIBE) {
                if (m.contains("active")) {
                    // for long polling, the subscription should stay active, if and only if the messages's "active" member is truthy
                    if (std::get<bool>(m.at("active"))) {
                        if (!_thread) {
                            startThread();
                        }
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

static_assert(gr::BlockLike<http::HttpBlock<uint8_t>>);

} // namespace gr::http

ENABLE_REFLECTION_FOR_TEMPLATE(gr::http::HttpBlock, out, url, endpoint, type, parameters);
auto registerHttpBlock = gr::registerBlock<gr::http::HttpBlock, float, double>(gr::globalBlockRegistry());

#endif // GNURADIO_HTTP_BLOCK_HPP
