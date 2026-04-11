#include <boost/ut.hpp>

#include <atomic>
#include <thread>
#include <vector>

#include <gnuradio-4.0/execution/execution.hpp>
#include <gnuradio-4.0/execution/pool_scheduler.hpp>

using namespace boost::ut;
using namespace std::string_view_literals;
namespace ex = gr::execution;

const suite<"gr::execution"> tests =
    [] {
        "just produces value via sync_wait"_test = [] {
            auto r = ex::sync_wait(ex::just(42));
            expect(r.has_value());
            expect(eq(std::get<0>(*r), 42));
        };

        "just void completes"_test = [] {
            auto r = ex::sync_wait(ex::just());
            expect(r.has_value()); // optional<tuple<>> with value
        };

        "then transforms value"_test = [] {
            auto r = ex::sync_wait(ex::just(3) | ex::then([](int x) { return x * 2; }));
            expect(r.has_value());
            expect(eq(std::get<0>(*r), 6));
        };

        "then chain of three stages"_test = [] {
            auto r = ex::sync_wait(ex::just(1) | ex::then([](int x) { return x + 1; }) | ex::then([](int x) { return x * 10; }) | ex::then([](int x) { return x + 5; }));
            expect(r.has_value());
            expect(eq(std::get<0>(*r), 25)); // (1+1)*10+5
        };

        "then with void input"_test = [] {
            auto r = ex::sync_wait(ex::just() | ex::then([] { return 99; }));
            expect(r.has_value());
            expect(eq(std::get<0>(*r), 99));
        };

        "then void to void"_test = [] {
            bool called = false;
            auto r      = ex::sync_wait(ex::just() | ex::then([&called] { called = true; }));
            expect(r.has_value());
            expect(called);
        };

        "let_value chains senders"_test = [] {
            auto r = ex::sync_wait(ex::just(5) | ex::let_value([](int x) { return ex::just(x * 100); }));
            expect(r.has_value());
            expect(eq(std::get<0>(*r), 500));
        };

        "let_value with nested then"_test = [] {
            auto r = ex::sync_wait(ex::just(3) | ex::let_value([](int x) { return ex::just(x) | ex::then([](int v) { return v * v; }); }));
            expect(r.has_value());
            expect(eq(std::get<0>(*r), 9));
        };

        "when_all joins void senders"_test = [] {
            auto r = ex::sync_wait(ex::when_all(ex::just(), ex::just(), ex::just()));
            expect(r.has_value());
        };

        "bulk invokes N times"_test = [] {
            int  sum = 0;
            auto r   = ex::sync_wait(ex::just() | ex::bulk(std::size_t(10), [&sum](std::size_t i) { sum += static_cast<int>(i); }));
            expect(r.has_value());
            expect(eq(sum, 45)); // 0+1+...+9
        };

        "bulk with value passthrough"_test = [] {
            std::vector<int> vec(8, 0);
            auto             r = ex::sync_wait(ex::just(std::move(vec)) | ex::bulk(std::size_t(8), [](std::size_t i, std::vector<int>& v) { v[i] = static_cast<int>(i * i); }));
            expect(r.has_value());
            const auto& v = std::get<0>(*r);
            expect(eq(v[0], 0) and eq(v[1], 1) and eq(v[4], 16) and eq(v[7], 49));
        };

        "pipe syntax produces correct result"_test = [] {
            auto r = ex::sync_wait(ex::just(10) | ex::then([](int x) { return x + 5; }) | ex::then([](int x) { return x * 2; }));
            expect(r.has_value());
            expect(eq(std::get<0>(*r), 30)); // (10+5)*2
        };

        "schedule runs on pool thread"_test = [] {
            auto                         sched = ex::cpuScheduler();
            std::atomic<std::thread::id> poolThreadId{};

            auto r = ex::sync_wait(sched.schedule() | ex::then([&poolThreadId] { poolThreadId.store(std::this_thread::get_id()); }));
            expect(r.has_value());
            expect(poolThreadId.load() != std::thread::id{});
        };

        "continues_on hops between pools"_test = [] {
            auto cpu = ex::cpuScheduler();
            auto io  = ex::ioScheduler();

            std::atomic<std::thread::id> cpuThreadId{};
            std::atomic<std::thread::id> ioThreadId{};

            auto r = ex::sync_wait(cpu.schedule() | ex::then([&cpuThreadId] { cpuThreadId.store(std::this_thread::get_id()); }) | ex::continues_on(io) | ex::then([&ioThreadId] { ioThreadId.store(std::this_thread::get_id()); }));
            expect(r.has_value());
            expect(cpuThreadId.load() != std::thread::id{});
            expect(ioThreadId.load() != std::thread::id{});
        };

        "schedule + then + sync_wait round-trip"_test = [] {
            auto sched = ex::cpuScheduler();
            auto r     = ex::sync_wait(sched.schedule() | ex::then([] { return 42; }));
            expect(r.has_value());
            expect(eq(std::get<0>(*r), 42));
        };

        "when_all with concurrent pool tasks"_test = [] {
            auto cpu = ex::cpuScheduler();

            std::atomic<int> counter{0};
            auto             s1 = cpu.schedule() | ex::then([&counter] { counter.fetch_add(1); });
            auto             s2 = cpu.schedule() | ex::then([&counter] { counter.fetch_add(10); });
            auto             s3 = cpu.schedule() | ex::then([&counter] { counter.fetch_add(100); });

            auto r = ex::sync_wait(ex::when_all(std::move(s1), std::move(s2), std::move(s3)));
            expect(r.has_value());
            expect(eq(counter.load(), 111));
        };

        "error propagation through then"_test = [] {
            bool caught = false;
            try {
                [[maybe_unused]] auto r = ex::sync_wait(ex::just(0) | ex::then([](int) -> int { throw std::runtime_error("test error"); }));
            } catch (const std::runtime_error& e) {
                caught = true;
                expect(std::string_view(e.what()) == "test error"sv);
            }
            expect(caught);
        };
};

int main() { /* not needed for UT */ }
