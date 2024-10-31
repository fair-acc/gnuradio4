#include <boost/ut.hpp>

#include <gnuradio-4.0/thread/thread_pool.hpp>

const boost::ut::suite ThreadPoolTests = [] {
    using namespace boost::ut;

    "Basic ThreadPool tests"_test = [] {
        expect(nothrow([] { gr::thread_pool::BasicThreadPool("test", gr::thread_pool::IO_BOUND, 4UL); }));
        expect(nothrow([] { gr::thread_pool::BasicThreadPool("test2", gr::thread_pool::CPU_BOUND, 4UL); }));

        std::atomic<int>                 enqueueCount{0};
        std::atomic<int>                 executeCount{0};
        gr::thread_pool::BasicThreadPool pool("TestPool", gr::thread_pool::IO_BOUND, 1, 2);
        expect(nothrow([&] { pool.sleepDuration = std::chrono::milliseconds(1); }));
        expect(nothrow([&] { pool.keepAliveDuration = std::chrono::seconds(10); }));
        pool.waitUntilInitialised();
        expect(that % pool.isInitialised());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        expect(pool.poolName() == "TestPool");
        expect(pool.minThreads() == 1_u);
        expect(pool.maxThreads() == 2_u);
        expect(pool.numThreads() == 1_u);
        expect(pool.numTasksRunning() == 0_u);
        expect(pool.numTasksQueued() == 0_u);
        expect(pool.numTasksRecycled() == 0_u);
        pool.execute([&enqueueCount] {
            ++enqueueCount;
            enqueueCount.notify_all();
        });
        enqueueCount.wait(0);
        expect(pool.numThreads() == 1_u);
        pool.execute([&executeCount] {
            ++executeCount;
            executeCount.notify_all();
        });
        executeCount.wait(0);
        expect(pool.numThreads() >= 1_u);
        expect(enqueueCount.load() == 1_i);
        expect(executeCount.load() == 1_i);

        auto ret = pool.execute([] { return 42; });
        expect(ret.get() == 42_i);

        auto taskName = pool.execute<"taskName", 0, -1>([] { return gr::thread_pool::thread::getThreadName(); });
#if defined(__EMSCRIPTEN__) || defined(__APPLE__)
        expect(taskName.get() == "unknown thread name"_b);
#else
        expect(taskName.get() == "taskName"_b);
#endif

        expect(nothrow([&] { pool.setAffinityMask(pool.getAffinityMask()); }));
        expect(nothrow([&] { pool.setThreadSchedulingPolicy(pool.getSchedulingPolicy(), pool.getSchedulingPriority()); }));
    };
    "contention tests"_test = [] {
        std::atomic<int>                 counter{0};
        gr::thread_pool::BasicThreadPool pool("contention", gr::thread_pool::IO_BOUND, 1, 4);
        pool.waitUntilInitialised();
        expect(that % pool.isInitialised());
        expect(pool.numThreads() == 1_u);
        pool.execute([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::atomic_fetch_add(&counter, 1);
            counter.notify_all();
        });
        expect(pool.numThreads() == 1_u);
        pool.execute([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::atomic_fetch_add(&counter, 1);
            counter.notify_all();
        });
        expect(pool.numThreads() >= 1_u);
        counter.wait(0);
        counter.wait(1);
        expect(counter.load() == 2_i);
    };

    "ThreadPool: Thread count tests"_test = [] {
        struct bounds_def {
            std::uint32_t min, max;
        };
        std::array<bounds_def, 5> bounds{bounds_def{1, 1}, bounds_def{1, 4}, bounds_def{2, 2}, bounds_def{2, 8}, bounds_def{4, 8}};

        for (const auto [minThreads, maxThreads] : bounds) {
            for (const auto taskCount : {2, 8, 32}) {
                fmt::print("## Test with min={} and max={} and taskCount={}\n", minThreads, maxThreads, taskCount);
                std::atomic<int> counter{0};

                // Pool with min and max thread count
                gr::thread_pool::BasicThreadPool pool("count_test", gr::thread_pool::IO_BOUND, minThreads, maxThreads);
                pool.keepAliveDuration = std::chrono::milliseconds(10); // default is 10 seconds, reducing for testing
                pool.waitUntilInitialised();

                for (int i = 0; i < taskCount; ++i) {
                    pool.execute([&counter] {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        std::atomic_fetch_add(&counter, 1);
                        counter.notify_all();
                    });
                }
                expect(that % pool.numThreads() >= minThreads);
                // the maximum number of threads is not a hard limit, if there is a burst of execute calls, it will spwawn more than maxThreads trheads.
                // expect(that % pool.numThreads() == std::min(std::uint32_t(taskCount), maxThreads));

                for (int i = 0; i < taskCount; ++i) {
                    counter.wait(i);
                    expect(that % pool.numThreads() >= minThreads);
                    // expect(that % pool.numThreads() <= maxThreads); // not a hard limit
                }

                // We should have gotten back to minimum
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                expect(that % pool.numThreads() == minThreads);
                expect(that % counter.load() == taskCount);
            }
        }
    };
};

int main() { /* tests are statically executed */ }
