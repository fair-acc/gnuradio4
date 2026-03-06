#include <boost/ut.hpp>

#include <atomic>
#include <chrono>
#include <format>
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

#include <magic_enum.hpp>

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

[[nodiscard]] gr::Tensor<std::uint8_t> byteTensor(std::string_view value) { return gr::Tensor<std::uint8_t>(value.begin(), value.end()); }

template<typename Condition>
bool awaitCondition(std::chrono::milliseconds timeout, Condition condition) {
    using namespace std::chrono_literals;
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < timeout) {
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(1ms);
    }
    return false;
}

const boost::ut::suite HttpBlocktests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace std::literals;
    using namespace std::chrono_literals;

#ifdef __EMSCRIPTEN__
    std::thread emscriptenThread{[]() {
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
#ifdef __EMSCRIPTEN__
        httpSource._emscriptenRunOnMainThread = false;
#endif
        expect(graph.connect<"out", "in">(httpSource, sink).has_value());

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
#ifdef __EMSCRIPTEN__
        httpSource._emscriptenRunOnMainThread = false;
#endif
        expect(graph.connect<"out", "in">(httpSource, sink).has_value());

        gr::MsgPortIn           fromScheduler;
        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.msgOut.connect(fromScheduler).has_value());
        expect(sched.runAndWait().has_value());
        expect(eq(sched.state(), lifecycle::State::ERROR));

#ifndef __EMSCRIPTEN__
        server.stop();
        thread.join();
#endif
    };

    "http POST"_test = [] {
#ifndef __EMSCRIPTEN__
        std::atomic_bool received = false;
        std::string      receivedBody;
        httplib::Server  server;
        server.Post("/number", [&](const httplib::Request& req, httplib::Response& res) {
            if (!received.exchange(true)) {
                receivedBody = req.body;
            }
            res.set_content("OK", "text/plain");
        });

        auto thread = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();
#endif

        gr::Graph  graph;
        const auto payload   = byteTensor("param=42");
        const auto chunkSize = payload.size();
        auto&      source    = graph.emplaceBlock<gr::testing::TagSource<std::uint8_t, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"values", payload}, {"n_samples_max", gr::Size_t(0)}});
        auto&      sink      = graph.emplaceBlock<gr::http::HttpSink>({{"url", gr::pmt::Value("http://localhost:8080/number")}, {"content_type", gr::pmt::Value("application/x-www-form-urlencoded")}});
#ifdef __EMSCRIPTEN__
        sink._emscriptenRunOnMainThread = false;
#endif
        expect(graph.connect<"out", "in">(source, sink).has_value());
        sink.in.max_samples = chunkSize;

        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.changeStateTo(lifecycle::State::INITIALISED).has_value());
        expect(sched.changeStateTo(lifecycle::State::RUNNING).has_value());
        expect(awaitCondition(2s, [&sched] { return sched.state() == lifecycle::State::RUNNING; })) << "scheduler did not enter RUNNING state";

#ifndef __EMSCRIPTEN__
        expect(awaitCondition(2s, [&received] { return received.load(); })) << "POST request was not received";
        expect(eq(receivedBody, "param=42"s));
#endif

        if (sched.state() != lifecycle::State::STOPPED) {
            const auto lifeStateChangeResult = sched.changeStateTo(lifecycle::State::REQUESTED_STOP);
            expect(lifeStateChangeResult.has_value()) << [&lifeStateChangeResult]() { return std::format("failed to set REQUESTED_STOP: {} at {}", lifeStateChangeResult.error().message, lifeStateChangeResult.error().sourceLocation); };
            expect(awaitCondition(2s, [&sched] { return sched.state() == lifecycle::State::STOPPED; })) << std::format("scheduler should be stopped - actual: {}", magic_enum::enum_name(sched.state()));
        }
#ifndef __EMSCRIPTEN__
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
#ifdef __EMSCRIPTEN__
        httpSource._emscriptenRunOnMainThread = false;
#endif
        expect(graph.connect<"out", "in">(httpSource, sink).has_value());

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
