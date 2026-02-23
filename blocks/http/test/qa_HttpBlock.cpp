#include <future>

#include <boost/ut.hpp>

#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <gnuradio-4.0/http/HttpBlock.hpp>

#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

static_assert(gr::BlockLike<http::HttpBlock<uint8_t>>);

template<typename T>
class FixedSource : public gr::Block<FixedSource<T>> {
    using super_t = gr::Block<FixedSource<T>>;

public:
    gr::PortOut<T> out;
    T              value = 1;

    GR_MAKE_REFLECTABLE(FixedSource, out);

    [[nodiscard]] constexpr auto processOne() noexcept { return value; }

    void trigger() { super_t::emitMessage("custom_kind", {}); }
};

template<typename T>
class HttpTestSink : public gr::Block<HttpTestSink<T>> {
public:
    gr::PortIn<T> in;

    GR_MAKE_REFLECTABLE(HttpTestSink, in);

    T                     value{};
    std::function<void()> stopFunc;

    constexpr void processOne(T val) {
        value = val;
        if (!value.empty()) {
            stopFunc();
        }
    }

    void reset() { value = {}; }
};

const boost::ut::suite HttpBlocktests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace std::literals;
    using namespace std::chrono_literals;

#ifdef __EMSCRIPTEN__
    std::thread emscriptenThread{[&]() {
        // see ./pre.js for the emscripten server implementation
        emscripten_run_script("startServer();");
    }};
#endif

    "http GET"_test = [&] {
#ifndef __EMSCRIPTEN__
        httplib::Server server;
        server.Get("/echo", [](const httplib::Request, httplib::Response& res) { res.set_content("Hello world!", "text/plain"); });

        auto thread = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();
#endif

        gr::Graph graph;
        auto&     source    = graph.emplaceBlock<FixedSource<uint8_t>>();
        auto&     httpBlock = graph.emplaceBlock<http::HttpBlock<uint8_t>>({{"url", gr::pmt::Value("http://localhost:8080")}, {"endpoint", gr::pmt::Value("/echo")}});

        auto& sink = graph.emplaceBlock<HttpTestSink<pmt::Value::Map>>();

        expect(eq(ConnectionResult::SUCCESS, source.msgOut.connect(httpBlock.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out", "in">(httpBlock, sink)));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        sink.stopFunc = [&]() { expect(sched.changeStateTo(lifecycle::State::REQUESTED_STOP).has_value()); };
        // make a request
        source.trigger();
        httpBlock.processScheduledMessages();
        expect(sched.runAndWait().has_value());
        expect(eq(gr::test::get_value_or_fail<std::string>(sink.value.at("raw-data")), std::string("Hello world!")));

#ifndef __EMSCRIPTEN__
        server.stop();
        thread.join();
#endif
    };

    "http GET 404"_test = [&] {
#ifndef __EMSCRIPTEN__
        httplib::Server server;
        server.Get("/does-not-exist", [](const httplib::Request, httplib::Response& res) { res.status = httplib::StatusCode::NotFound_404; });

        auto thread = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();
#endif

        gr::Graph graph;
        auto&     source    = graph.emplaceBlock<FixedSource<uint8_t>>();
        auto&     httpBlock = graph.emplaceBlock<http::HttpBlock<uint8_t>>(property_map{{"url", gr::pmt::Value("http://localhost:8080")}, {"endpoint", gr::pmt::Value("/does-not-exist")}});
        auto&     sink      = graph.emplaceBlock<HttpTestSink<pmt::Value::Map>>();

        expect(eq(ConnectionResult::SUCCESS, source.msgOut.connect(httpBlock.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out", "in">(httpBlock, sink)));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        sink.stopFunc = [&]() { expect(sched.changeStateTo(lifecycle::State::REQUESTED_STOP).has_value()); };
        httpBlock.trigger();
        expect(sched.runAndWait().has_value());
        expect(eq(gr::test::get_value_or_fail<int>(sink.value.at("status")), 404));

#ifndef __EMSCRIPTEN__
        server.stop();
        thread.join();
#endif
    };

    "http POST"_test = [] {
#ifndef __EMSCRIPTEN__
        httplib::Server server;
        server.Post("/number", [](const httplib::Request&, httplib::Response& res) { res.set_content("OK", "text/plain"); });

        auto thread = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();
#endif

        gr::Graph graph;
        auto&     source    = graph.emplaceBlock<FixedSource<uint8_t>>();
        auto&     httpBlock = graph.emplaceBlock<http::HttpBlock<uint8_t>>({{"url", gr::pmt::Value("http://localhost:8080")}, {"endpoint", gr::pmt::Value("/number")}, {"type", gr::pmt::Value("POST")}, {"parameters", gr::pmt::Value("param=42")}});
        auto&     sink      = graph.emplaceBlock<HttpTestSink<pmt::Value::Map>>();

        expect(eq(ConnectionResult::SUCCESS, source.msgOut.connect(httpBlock.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out", "in">(httpBlock, sink)));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        sink.stopFunc = [&]() { expect(sched.changeStateTo(lifecycle::State::REQUESTED_STOP).has_value()); };
        httpBlock.trigger();
        expect(sched.runAndWait().has_value());
        expect(eq(gr::test::get_value_or_fail<std::string>(sink.value.at("raw-data")), "OK"sv));

#ifndef __EMSCRIPTEN__
        server.stop();
        thread.join();
#endif
    };

    "http SUBSCRIBE"_test = [] {
#ifndef __EMSCRIPTEN__
        std::atomic_bool shutdown = false;
        httplib::Server  server;
        server.Get("/notify", [&](const httplib::Request, httplib::Response& res) {
            res.set_chunked_content_provider("text/plain", [&](size_t, httplib::DataSink& sink) {
                if (shutdown) {
                    return false;
                }
                // delay the reply a bit, we are long polling
                std::this_thread::sleep_for(10ms);
                if (sink.is_writable()) {
                    sink.os << "event";
                }
                return true;
            });
        });

        auto thread = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();
#endif

        gr::Graph graph;
        auto&     source    = graph.emplaceBlock<FixedSource<uint8_t>>();
        auto&     httpBlock = graph.emplaceBlock<http::HttpBlock<uint8_t>>({{"url", gr::pmt::Value("http://localhost:8080")}, {"endpoint", gr::pmt::Value("/notify")}, {"type", gr::pmt::Value("SUBSCRIBE")}});
        auto&     sink      = graph.emplaceBlock<HttpTestSink<pmt::Value::Map>>();

        expect(eq(ConnectionResult::SUCCESS, source.msgOut.connect(httpBlock.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out", "in">(httpBlock, sink)));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        sink.stopFunc = [&]() { expect(sched.changeStateTo(lifecycle::State::REQUESTED_STOP).has_value()); };
        expect(sched.runAndWait().has_value());
        expect(eq(gr::test::get_value_or_fail<std::string>(sink.value.at("raw-data")), "event"sv));

#ifndef __EMSCRIPTEN__
        shutdown = true;
        server.stop();
        thread.join();
#endif
    };

#ifdef __EMSCRIPTEN__
    emscripten_run_script("stopServer();");
    emscriptenThread.join();
#endif
};

int main() { /* tests are statically executed */ return boost::ut::cfg<boost::ut::override>.run(); }
