#include <boost/ut.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <expected>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/testing/ExampleRun.hpp>

namespace {

const boost::ut::suite<"ExampleArguments"> argumentTests = [] {
    using namespace boost::ut;
    "flags, texts and numbers are consumed by name; the unknown, the missing value and the malformed number invalidate the line"_test = [] {
        const auto parse = [](std::vector<std::string> words) {
            std::vector<char*> argv;
            for (std::string& word : words) {
                argv.push_back(word.data());
            }
            bool                          synthetic = false;
            std::string                   device;
            double                        seconds = 0.0;
            std::size_t                   count   = 4UZ;
            gr::testing::ExampleArguments args(static_cast<int>(argv.size()), argv.data());
            while (args.next()) {
                if (!(args.flag("--synthetic", synthetic) || args.text("--device", device) || args.number("--seconds", seconds) || args.number("--count", count))) {
                    args.refuse();
                }
            }
            return std::tuple{args.valid, synthetic, device, seconds, count};
        };
        expect(parse({"prog", "--synthetic", "--device", "hw:1", "--seconds", "2.5", "--count", "8"}) == std::tuple{true, true, std::string("hw:1"), 2.5, 8UZ});
        expect(parse({"prog"}) == std::tuple{true, false, std::string(), 0.0, 4UZ}) << "no arguments: the defaults";
        expect(!std::get<0>(parse({"prog", "--bogus"}))) << "an unknown flag";
        expect(!std::get<0>(parse({"prog", "--seconds"}))) << "a flag without its value";
        expect(!std::get<0>(parse({"prog", "--seconds", "soon"}))) << "a malformed number";
        expect(!std::get<0>(parse({"prog", "--seconds", "inf"}))) << "a non-finite number";
        expect(!std::get<0>(parse({"prog", "--count", "3x"}))) << "a number with trailing text";
    };
};

enum class Outcome { complete, failure, deviceFault, devicePoisoned, deadline, deadlineBeforeRunning, interruptBeforeRunning, interrupt, terminate };

struct ExampleScheduler {
    Outcome                           outcome;
    std::atomic<gr::lifecycle::State> currentState{gr::lifecycle::State::INITIALISED};
    std::atomic<bool>                 stopped{false};

    auto state() const { return currentState.load(); }
    void requestStop() { stopped = true; }

    std::expected<void, gr::Error> exchange(gr::Graph&&) { return {}; }

    std::expected<void, gr::Error> runAndWait() {
        if (outcome == Outcome::interruptBeforeRunning) {
            std::raise(SIGINT);
        }
        if (outcome == Outcome::deadlineBeforeRunning) {
            const auto startupDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(30);
            while (std::chrono::steady_clock::now() < startupDeadline) {
                std::this_thread::yield();
            }
        }
        currentState = gr::lifecycle::State::RUNNING;
        if (outcome == Outcome::complete) {
            return {};
        }
        if (outcome == Outcome::failure) {
            return std::unexpected(gr::Error("test graph failure"));
        }
        if (outcome == Outcome::deviceFault) {
            return std::unexpected(gr::Error("device fault during processBulk dispatch: JIT link failed"));
        }
        if (outcome == Outcome::devicePoisoned) {
            return std::unexpected(gr::Error("device context poisoned by a prior error: JIT link failed"));
        }
        if (outcome == Outcome::interrupt) {
            std::raise(SIGINT);
        } else if (outcome == Outcome::terminate) {
            std::raise(SIGTERM);
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!stopped && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        if (!stopped) {
            return std::unexpected(gr::Error("example stop request was not observed within two seconds"));
        }
        currentState = gr::lifecycle::State::STOPPED;
        return {};
    }
};

std::atomic_flag priorHandlerCalled{};
void             priorHandler(int) { priorHandlerCalled.test_and_set(); }

const boost::ut::suite<"ExampleRun"> runTests = [] {
    using namespace boost::ut;

    "natural completion and graph errors propagate without a time limit"_test = [] {
        ExampleScheduler complete{.outcome = Outcome::complete};
        expect(gr::testing::runExample(complete, 0.0).has_value());
        ExampleScheduler failure{.outcome = Outcome::failure};
        const auto       result = gr::testing::runExample(failure, 0.0);
        expect(fatal(!result.has_value()));
        expect(eq(result.error().message, std::string("test graph failure")));
    };

    "interrupts during startup and running stop gracefully and restore prior handlers"_test = [] {
        const auto              oldInterrupt = std::signal(SIGINT, priorHandler);
        const auto              oldTerminate = std::signal(SIGTERM, priorHandler);
        const gr::on_scope_exit restore([&] {
            std::signal(SIGINT, oldInterrupt);
            std::signal(SIGTERM, oldTerminate);
        });
        for (const auto outcome : {Outcome::interruptBeforeRunning, Outcome::interrupt, Outcome::terminate}) {
            ExampleScheduler scheduler{.outcome = outcome};
            expect(gr::testing::runExample(scheduler, 0.0).has_value());
            expect(scheduler.stopped.load());
            priorHandlerCalled.clear();
            std::raise(SIGINT);
            expect(priorHandlerCalled.test());
            priorHandlerCalled.clear();
            std::raise(SIGTERM);
            expect(priorHandlerCalled.test());
        }
        ExampleScheduler complete{.outcome = Outcome::complete};
        expect(gr::testing::runExample(complete, 0.0).has_value());
        expect(!gr::testing::detail::exampleInterrupted.test());
    };

    "a positive duration stops a running scheduler without a signal"_test = [] {
        for (const auto outcome : {Outcome::deadline, Outcome::deadlineBeforeRunning}) {
            ExampleScheduler scheduler{.outcome = outcome};
            expect(gr::testing::runExample(scheduler, 0.001).has_value());
            expect(scheduler.stopped.load());
            expect(!gr::testing::detail::exampleInterrupted.test());
        }
    };

    "a device fault or a context it poisoned on the GPU domain rebuilds the graph on the host and runs it again"_test = [] {
        for (const Outcome fault : {Outcome::deviceFault, Outcome::devicePoisoned}) {
            ExampleScheduler              scheduler{.outcome = Outcome::complete};
            std::vector<std::string_view> domains;
            const auto                    buildGraph = [&](std::string_view domain) -> std::expected<gr::Graph, std::string> {
                domains.push_back(domain);
                scheduler.outcome = domain == gr::testing::kExampleDeviceDomain ? fault : Outcome::complete;
                return gr::Graph();
            };
            expect(eq(gr::testing::runExample(buildGraph, gr::testing::kExampleDeviceDomain, 0.0, scheduler), 0));
            expect(domains == std::vector{gr::testing::kExampleDeviceDomain, gr::thread_pool::kDefaultIoPoolId}) << "the GPU first, then the host";
        }
    };

    "a failure other than a device fault, or a device fault on the host, ends the example without a retry"_test = [] {
        const auto runOnce = [](Outcome outcome, std::string_view domain) {
            ExampleScheduler              scheduler{.outcome = outcome};
            std::vector<std::string_view> domains;
            const auto                    buildGraph = [&](std::string_view requested) -> std::expected<gr::Graph, std::string> {
                domains.push_back(requested);
                return gr::Graph();
            };
            const int exitCode = gr::testing::runExample(buildGraph, domain, 0.0, scheduler);
            return std::tuple{exitCode, domains};
        };
        expect(runOnce(Outcome::failure, gr::testing::kExampleDeviceDomain) == std::tuple{1, std::vector{gr::testing::kExampleDeviceDomain}}) << "a graph failure on the GPU";
        expect(runOnce(Outcome::deviceFault, gr::thread_pool::kDefaultIoPoolId) == std::tuple{1, std::vector{gr::thread_pool::kDefaultIoPoolId}}) << "a device fault under --nogpu";
    };

    "a graph that cannot be built ends the example before it runs"_test = [] {
        ExampleScheduler scheduler{.outcome = Outcome::deviceFault};
        const auto       buildGraph = [](std::string_view) -> std::expected<gr::Graph, std::string> { return std::unexpected(std::string("microphone missing")); };
        expect(eq(gr::testing::runExample(buildGraph, gr::testing::kExampleDeviceDomain, 0.0, scheduler), 1));
        expect(scheduler.currentState.load() == gr::lifecycle::State::INITIALISED) << "the scheduler never ran";
    };

    "the correlation frame is 1024 samples up to a lag of 511 and doubles once the half frame no longer exceeds the lag"_test = [] {
        expect(eq(gr::testing::correlationFrameSamples(0UZ), 1024UZ));
        const auto frame = gr::testing::correlationFrameFor(3.0, 343.0, 48000.0); // 420 lags plus two: a half frame of 512 holds them
        expect(fatal(frame.has_value()));
        expect(eq(frame->maxLag, 422U) && eq(frame->samples, 1024U));
        expect(eq(gr::testing::correlationFrameFor(8.0, 343.0, 48000.0)->samples, 4096U)) << "1120 lags need a half frame of 2048";
        expect(!gr::testing::correlationFrameFor(1000.0, 343.0, 48000.0).has_value()) << "a kilometre of aperture exceeds the largest frame";
        expect(!gr::testing::correlationFrameFor(std::numeric_limits<double>::infinity(), 343.0, 48000.0).has_value()) << "an infinite aperture is refused, not cast";
        expect(eq(gr::testing::correlationFrameSamples(511UZ), 1024UZ));
        expect(eq(gr::testing::correlationFrameSamples(512UZ), 2048UZ));
        expect(eq(gr::testing::correlationFrameSamples(842UZ), 2048UZ));
        expect(eq(gr::testing::correlationFrameSamples(1023UZ), 2048UZ));
        expect(eq(gr::testing::correlationFrameSamples(1024UZ), 4096UZ));
        static_assert(gr::testing::correlationFrameSamples(300UZ) == 1024UZ);
    };

    "a device's channel count sets the default array: 2 a stereo line 0.3 m apart, 4 a tetrahedron of 0.3 m edge, 9 a square grid and 16 the UMA-16 at 42 mm, 8 a line 1 m apart"_test = [] {
        const auto expectArray = [](std::size_t channels, std::string_view layout, std::size_t rows, std::size_t columns, double spacing) {
            const gr::testing::ChannelArray array = gr::testing::channelArrayOf(channels);
            expect(array.layout == layout && eq(array.elements, channels) && eq(array.rows, rows) && eq(array.columns, columns) && eq(array.spacing, spacing)) << std::format("{} channels", channels);
        };
        expectArray(2UZ, "line", 1UZ, 2UZ, 0.3);
        expectArray(4UZ, "tetrahedron", 1UZ, 4UZ, 0.3);
        expectArray(9UZ, "grid", 3UZ, 3UZ, 0.042);
        expectArray(16UZ, "uma16", 4UZ, 4UZ, 0.042);
        expectArray(8UZ, "line", 1UZ, 8UZ, 1.0);
    };
};

} // namespace

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
