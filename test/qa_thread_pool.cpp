#include <boost/ut.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif
#include <thread_pool.hpp>

const boost::ut::suite ThreadPoolTests = [] {
    using namespace boost::ut;

    "Basic ThreadPool tests"_test = [] {
        expect(nothrow([]{ fair::thread_pool::BasicThreadPool<fair::thread_pool::IO_BOUND>(); }));
        expect(nothrow([]{ fair::thread_pool::BasicThreadPool<fair::thread_pool::CPU_BOUND>(); }));

        std::atomic<int>                            enqueueCount{ 0 };
        std::atomic<int>                            executeCount{ 0 };
        fair::thread_pool::BasicThreadPool<fair::thread_pool::IO_BOUND> pool("TestPool", 1, 2);
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
        pool.execute([&enqueueCount] { ++enqueueCount; enqueueCount.notify_all(); });
        enqueueCount.wait(0);
        expect(pool.numThreads() == 1_u);
        pool.execute([&executeCount] { ++executeCount; executeCount.notify_all(); });
        executeCount.wait(0);
        expect(pool.numThreads() >= 1_u);
        expect(enqueueCount.load() == 1_i);
        expect(executeCount.load() == 1_i);

        auto ret = pool.execute([] { return 42; });
        expect(ret.get() == 42_i);

        auto taskName = pool.execute<"taskName", 0, -1>([] { return fair::thread_pool::thread::getThreadName(); });
#ifdef __EMSCRIPTEN__
        expect(taskName.get() == "unknown thread name"_b);
#else
        expect(taskName.get() == "taskName"_b);
#endif

        expect(nothrow([&]{ pool.setAffinityMask(pool.getAffinityMask()); }));
        expect(nothrow([&]{ pool.setThreadSchedulingPolicy(pool.getSchedulingPolicy(), pool.getSchedulingPriority()); }));
    };
    "contention tests"_test = [] {
        std::atomic<int>                            counter{ 0 };
        fair::thread_pool::BasicThreadPool<fair::thread_pool::IO_BOUND> pool("contention", 1, 4);
        pool.waitUntilInitialised();
        expect(that % pool.isInitialised());
        expect(pool.numThreads() == 1_u);
        pool.execute([&counter] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); std::atomic_fetch_add(&counter, 1); counter.notify_all(); });
        expect(pool.numThreads() == 1_u);
        pool.execute([&counter] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); std::atomic_fetch_add(&counter, 1); counter.notify_all(); });
        expect(pool.numThreads() >= 1_u);
        counter.wait(0);
        counter.wait(1);
        expect(counter.load() == 2_i);
    };
};

int
main() { /* tests are statically executed */
}
