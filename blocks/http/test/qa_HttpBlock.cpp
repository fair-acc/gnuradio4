#include <future>

#include <boost/ut.hpp>

#include <atomic>
#include <chrono>
#include <format>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <gnuradio-4.0/http/HttpBlock.hpp>

#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#ifndef __EMSCRIPTEN__
#include <httplib.h>
#endif

static_assert(gr::BlockLike<gr::http::HttpSource>);
static_assert(gr::BlockLike<gr::http::HttpSink>);

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

using ByteTagSource = gr::testing::TagSource<std::uint8_t, gr::testing::ProcessFunction::USE_PROCESS_BULK>;

[[nodiscard]] gr::Tensor<std::uint8_t> byteTensor(std::string_view value) { return gr::Tensor<std::uint8_t>(value.begin(), value.end()); }

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
        server.Get("/echo", [](const httplib::Request&, httplib::Response& res) { res.set_content("Hello world!", "text/plain"); });

        auto thread = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();
#endif

        gr::Graph graph;
        auto&     httpSource = graph.emplaceBlock<gr::http::HttpSource>({{"url", gr::pmt::Value("http://localhost:8080/echo")}});
        auto&     sink       = graph.emplaceBlock<HttpTestSink<pmt::Value::Map>>();

        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(httpSource).template to<"in">(sink)));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        sink.stopFunc = [&]() { expect(sched.changeStateTo(lifecycle::State::REQUESTED_STOP).has_value()); };
        expect(sched.runAndWait().has_value());
        expect(eq(gr::test::get_value_or_fail<gr::Tensor<std::uint8_t>>(sink.value.at("raw-data")), byteTensor("Hello world!")));

#ifndef __EMSCRIPTEN__
        server.stop();
        thread.join();
#endif
    };

    "http GET 404"_test = [&] {
#ifndef __EMSCRIPTEN__
        httplib::Server server;
        server.Get("/does-not-exist", [](const httplib::Request&, httplib::Response& res) { res.status = httplib::StatusCode::NotFound_404; });

        auto thread = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();
#endif

        gr::Graph graph;
        auto&     httpSource = graph.emplaceBlock<gr::http::HttpSource>({{"url", gr::pmt::Value("http://localhost:8080/does-not-exist")}});
        auto&     sink       = graph.emplaceBlock<HttpTestSink<pmt::Value::Map>>();

        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(httpSource).template to<"in">(sink)));

        gr::MsgPortIn           fromScheduler;
        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(eq(ConnectionResult::SUCCESS, sched.msgOut.connect(fromScheduler)));
        expect(sched.runAndWait().has_value());
        expect(eq(sched.state(), lifecycle::State::ERROR));

#ifndef __EMSCRIPTEN__
        server.stop();
        thread.join();
#endif
    };

    "http POST"_test = [] {
#ifndef __EMSCRIPTEN__
        std::string     receivedBody;
        httplib::Server server;
        server.Post("/number", [&](const httplib::Request& req, httplib::Response& res) {
            receivedBody = req.body;
            res.set_content("OK", "text/plain");
        });

        auto thread = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();
#endif

        gr::Graph graph;
        auto&     source = graph.emplaceBlock<ByteTagSource>({{"values", byteTensor("param=42")}, {"n_samples_max", gr::Size_t(8)}});
        auto&     sink   = graph.emplaceBlock<gr::http::HttpSink>({{"url", gr::pmt::Value("http://localhost:8080/number")}, {"content_type", gr::pmt::Value("application/x-www-form-urlencoded")}});

        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(source).template to<"in">(sink)));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());

#ifndef __EMSCRIPTEN__
        expect(eq(receivedBody, "param=42"s));
        server.stop();
        thread.join();
#endif
    };

    "http SUBSCRIBE"_test = [] {
#ifndef __EMSCRIPTEN__
        std::atomic_bool shutdown = false;
        httplib::Server  server;
        server.Get("/notify", [&](const httplib::Request&, httplib::Response& res) {
            if (shutdown.load()) {
                res.status = httplib::StatusCode::ServiceUnavailable_503;
                return;
            }
            std::this_thread::sleep_for(10ms);
            res.set_content("event", "text/plain");
        });

        auto thread = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();
#endif

        gr::Graph graph;
        auto&     httpSource = graph.emplaceBlock<gr::http::HttpSource>({{"url", gr::pmt::Value("http://localhost:8080/notify")}, {"type", gr::pmt::Value("SUBSCRIBE")}});
        auto&     sink       = graph.emplaceBlock<HttpTestSink<pmt::Value::Map>>();

        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(httpSource).template to<"in">(sink)));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        sink.stopFunc = [&]() { expect(sched.changeStateTo(lifecycle::State::REQUESTED_STOP).has_value()); };
        expect(sched.runAndWait().has_value());
        expect(eq(gr::test::get_value_or_fail<gr::Tensor<std::uint8_t>>(sink.value.at("raw-data")), byteTensor("event")));

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
