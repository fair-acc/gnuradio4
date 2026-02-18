#ifndef GNURADIO_HTTP_BLOCK_HPP
#define GNURADIO_HTTP_BLOCK_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

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
struct HttpBlock : Block<HttpBlock<T>> {
    using Description = Doc<R""(
The HttpBlock allows to use the responses from HTTP APIs (e.g. REST APIs) as the value for this block's output port.
The block can be used either on-demand to do single requests, or can use long polling to subscribe to an event stream.
The result is provided on a single output port as a map with the following keys:
- status: The HTTP status code, usually 200 on success
- raw-data: The data of the response
- mime-type: The mime-type of the response
)"">;

    PortOut<pmt::Value::Map> out;

    gr::Annotated<std::string, "URI">                                                                  url;
    gr::Annotated<std::string, "endpoint">                                                             endpoint = "/";
    gr::Annotated<gr::http::RequestType, "type", gr::Doc<"GET, SUBSCRIBE, POST">>                      type     = gr::http::RequestType::GET;
    gr::Annotated<std::string, "parameters", gr::Doc<"x-www-form-urlencoded encoded POST parameters">> parameters;

    GR_MAKE_REFLECTABLE(HttpBlock, out, url, endpoint, type, parameters);

    std::atomic_size_t    _pendingRequests = 0UZ;
    std::atomic_bool      _shutdownThread  = false;
    std::binary_semaphore _ready{0};
    std::atomic_size_t    _samplesProduced = 0UZ;
    std::future<void>     _taskCompletion;

#ifndef __EMSCRIPTEN__
    std::unique_ptr<httplib::Client> _client;
#endif

    struct StopGuard { // must be last member — destroyed first, ensuring cleanup before other members
        HttpBlock* _owner = nullptr;
        ~StopGuard() {
            if (_owner) {
                _owner->stopThread();
            }
        }
    };
    StopGuard _stopGuard{this};

#ifdef __EMSCRIPTEN__
    void publishEmscriptenResult(emscripten_fetch_t* fetch) {
        pmt::Value::Map result;
        result["mime-type"] = "text/plain";
        result["status"]    = static_cast<int>(fetch->status);
        result["raw-data"]  = std::string(fetch->data, static_cast<std::size_t>(fetch->numBytes));

        publishResult(std::move(result));
    }

    void onSuccess(emscripten_fetch_t* fetch) {
        publishEmscriptenResult(fetch);
        emscripten_fetch_close(fetch);
    }

    void onError(emscripten_fetch_t* fetch) {
        // we still want to queue the response, the statusCode will just not be 200
        publishEmscriptenResult(fetch);
        emscripten_fetch_close(fetch);
    }

    void doRequestEmscripten() {
        emscripten_fetch_attr_t attr;
        emscripten_fetch_attr_init(&attr);
        if (type == RequestType::POST) {
            strcpy(attr.requestMethod, "POST");
            if (!parameters.value.empty()) {
                attr.requestData     = parameters.value.c_str();
                attr.requestDataSize = parameters.value.size();
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
        const auto target = url.value + endpoint.value;
        std::ignore       = emscripten_fetch(&attr, target.c_str());
    }

    void runThreadEmscripten() {
        if (type == RequestType::SUBSCRIBE) {
            while (!_shutdownThread) {
                // long polling, just keep doing requests
                std::thread thread{&HttpBlock::doRequestEmscripten, this};
                thread.join();
            }
        } else {
            while (!_shutdownThread) {
                while (_pendingRequests > 0) {
                    _pendingRequests--;
                    std::thread thread{&HttpBlock::doRequestEmscripten, this};
                    thread.join();
                }
                _ready.acquire();
            }
        }
    }
#else
    void runThreadNative() {
        _client = std::make_unique<httplib::Client>(url.value);
        _client->set_follow_location(true);
        if (type == RequestType::SUBSCRIBE) {
            _client->set_read_timeout(1h);
            _client->Get(endpoint.value, [&](const char* data, size_t len) {
                pmt::Value::Map result;
                result["mime-type"] = "text/plain";
                result["status"]    = 200;
                result["raw-data"]  = std::string(data, len);

                publishResult(std::move(result));

                return !_shutdownThread.load();
            });
        } else {
            while (!_shutdownThread) {
                while (_pendingRequests > 0) {
                    _pendingRequests--;
                    httplib::Result resp;
                    if (type == RequestType::POST) {
                        resp = parameters.value.empty() ? _client->Post(endpoint.value) : _client->Post(endpoint.value, parameters.value, "application/x-www-form-urlencoded");
                    } else {
                        resp = _client->Get(endpoint.value);
                    }
                    if (resp) {
                        pmt::Value::Map result;
                        result["mime-type"] = "text/plain";
                        result["status"]    = resp->status;
                        result["raw-data"]  = resp->body;
                        publishResult(std::move(result));
                    }
                }

                _ready.acquire();
            }
        }
    }
#endif

    void publishResult(pmt::Value::Map result) {
        while (!_shutdownThread.load(std::memory_order_relaxed)) {
            auto span = out.streamWriter().template tryReserve<SpanReleasePolicy::ProcessNone>(1UZ);
            if (!span.empty()) {
                span[0] = std::move(result);
                span.publish(1UZ);
                _samplesProduced.fetch_add(1UZ, std::memory_order_relaxed);
                this->progress->incrementAndGet();
                this->progress->notify_all();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void startThread() {
        stopThread();
        _shutdownThread = false;
        auto done       = std::make_shared<std::promise<void>>();
        _taskCompletion = done->get_future();
        thread_pool::Manager::defaultIoPool()->execute([this, done = std::move(done)]() mutable {
            thread_pool::thread::setThreadName(std::format("uT:{}", gr::meta::shorten_type_name(this->unique_name)));
#ifdef __EMSCRIPTEN__
            runThreadEmscripten();
#else
            runThreadNative();
#endif
            done->set_value();
        });
    }

    void stopThread() {
        if (!_taskCompletion.valid()) {
            return;
        }
        _shutdownThread = true;
        _ready.release();
#ifndef __EMSCRIPTEN__
        if (_client) {
            _client->stop();
        }
#endif
        _taskCompletion.wait();
        _taskCompletion = {};
    }

    void settingsChanged(const property_map& /*oldSettings*/, property_map& newSettings) {
        if (newSettings.contains("url") || newSettings.contains("type")) {
            if (_taskCompletion.valid()) {
                stopThread();
                startThread();
            }
        }
    }

    void start() { startThread(); }

    void stop() { stopThread(); }

    [[nodiscard]] constexpr auto processOne() noexcept { return pmt::Value::Map{}; }

    work::Result work(std::size_t requestedWork = std::numeric_limits<std::size_t>::max()) noexcept {
        this->applyChangedSettings();

        if (this->state() == lifecycle::State::REQUESTED_STOP) {
            this->emitErrorMessageIfAny("work(): REQUESTED_STOP -> STOPPED", this->changeStateTo(lifecycle::State::STOPPED));
        }
        if (!lifecycle::isActive(this->state())) {
            return {requestedWork, 0UZ, work::Status::DONE};
        }

        const auto produced = _samplesProduced.exchange(0UZ, std::memory_order_relaxed);
        return {requestedWork, produced, work::Status::OK};
    }

    void trigger() {
        _pendingRequests++;
        _ready.release();
    }

    void processMessages(gr::MsgPortInBuiltin& port, std::span<const gr::Message> message) {
        gr::Block<HttpBlock<T>>::processMessages(port, message);

        std::ranges::for_each(message, [this](auto& m) {
            if (type == RequestType::SUBSCRIBE) {
                if (m.data.has_value() && m.data.value().contains("active")) {
                    // for long polling, the subscription should stay active, if and only if the messages' "active" member is true
                    if (m.data.value().at("active").value_or(false)) {
                        if (!_taskCompletion.valid()) {
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

} // namespace gr::http

#endif // GNURADIO_HTTP_BLOCK_HPP
