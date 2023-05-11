#include <boost/ut.hpp>
#include <thread_pool.hpp>

const boost::ut::suite ThreadPoolTests = [] {
    using namespace boost::ut;

    "Basic ThreadPool tests"_test = [] {
        expect(nothrow([]{ fair::thread_pool::BasicThreadPool<fair::thread_pool::IO_BOUND>(); }));
        expect(nothrow([]{ fair::thread_pool::BasicThreadPool<fair::thread_pool::CPU_BOUND>(); }));

        std::atomic<int>                            enqueueCount{ 0 };
        std::atomic<int>                            executeCount{ 0 };
        fair::thread_pool::BasicThreadPool<fair::thread_pool::IO_BOUND> pool("TestPool", 1, 2);
        expect(nothrow([&]{ pool.sleepDuration() = std::chrono::milliseconds(1); }));
        expect(nothrow([&]{ pool.keepAliveDuration() = std::chrono::seconds(10); }));
        pool.waitUntilInitialised();
        expect(pool.isInitialised());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        expect(pool.poolName() == "TestPool");
        expect(pool.minThreads() == 1);
        expect(pool.maxThreads() == 2);
        expect(pool.numThreads() == 1);
        expect(pool.numTasksRunning() == 0);
        expect(pool.numTasksQueued() == 0);
        expect(pool.numTasksRecycled() == 0);
        pool.execute([&enqueueCount] { ++enqueueCount; enqueueCount.notify_all(); });
        enqueueCount.wait(0);
        expect(pool.numThreads() == 1);
        pool.execute([&executeCount] { ++executeCount; executeCount.notify_all(); });
        executeCount.wait(0);
        expect(pool.numThreads() >= 1);
        expect(enqueueCount == 1);
        expect(executeCount == 1);

        auto ret = pool.execute([] { return 42; });
        expect(ret.get() == 42);

        auto taskName = pool.execute<"taskName", 0, -1>([] { return fair::thread_pool::thread::getThreadName(); });
        expect(taskName.get() == "taskName");

        expect(nothrow([&]{ pool.setAffinityMask(pool.getAffinityMask()); }));
        expect(nothrow([&]{ pool.setThreadSchedulingPolicy(pool.getSchedulingPolicy(), pool.getSchedulingPriority()); }));
    };
    "contention tests"_test = [] {
        std::atomic<int>                            counter{ 0 };
        fair::thread_pool::BasicThreadPool<fair::thread_pool::IO_BOUND> pool("contention", 1, 4);
        pool.waitUntilInitialised();
        expect(pool.isInitialised());
        expect(pool.numThreads() == 1);
        pool.execute([&counter] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); std::atomic_fetch_add(&counter, 1); counter.notify_all(); });
        expect(pool.numThreads() == 1);
        pool.execute([&counter] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); std::atomic_fetch_add(&counter, 1); counter.notify_all(); });
        expect(pool.numThreads() >= 1);
        counter.wait(0);
        counter.wait(1);
        expect(counter == 2);
    };
};

int
main() { /* tests are statically executed */
}
