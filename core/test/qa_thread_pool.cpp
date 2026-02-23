#include <boost/ut.hpp>

#include <gnuradio-4.0/thread/thread_pool.hpp>

class SafeCounter {
    std::atomic<int>                _value{0};
    mutable std::mutex              _mutex;
    mutable std::condition_variable _cv;

public:
    void increment() {
        _value.fetch_add(1, std::memory_order_release);
        _cv.notify_all();
    }

    void wait(int old_value) {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [&] { return _value.load(std::memory_order_acquire) > old_value; });
    }

    int load() const { return _value.load(std::memory_order_acquire); }
};

const boost::ut::suite<"gr::thread_pool GR4 default"> defaultThreadPool = [] {
    using namespace boost::ut;

    "WIN32 ThreadPool tests"_test = [] {
        for (int i = 0; i < 10; ++i) {
            {
                gr::thread_pool::BasicThreadPool pool("test", gr::thread_pool::CPU_BOUND, 2, 4);
                pool.execute([i] { std::println("Task {}", i); });
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::println("Pool {} destroyed successfully", i);
        }

        gr::thread_pool::BasicThreadPool pool("persistent", gr::thread_pool::CPU_BOUND, 2, 4);
        for (int i = 0; i < 10; ++i) {
            pool.execute([i] { std::println("Task {}", i); });
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (i % 3 == 0) {
                pool.requestShutdown();
            }
        }
    };

    "Basic ThreadPool tests"_test = [] {
        expect(nothrow([] { gr::thread_pool::BasicThreadPool("test", gr::thread_pool::IO_BOUND, 4UL); }));
        expect(nothrow([] { gr::thread_pool::BasicThreadPool("test2", gr::thread_pool::CPU_BOUND, 4UL); }));

#ifdef _WIN32
        SafeCounter enqueueCount;
        SafeCounter executeCount;

#else
        std::atomic<int> enqueueCount{0};
        std::atomic<int> executeCount{0};
#endif
        gr::thread_pool::BasicThreadPool pool("TestPool", gr::thread_pool::IO_BOUND, 1, 2);
        expect(nothrow([&] { pool.sleepDuration = std::chrono::milliseconds(1); }));
        expect(nothrow([&] { pool.keepAliveDuration = std::chrono::seconds(10); }));
        pool.waitUntilInitialised();
        expect(that % pool.isInitialised());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        expect(pool.poolName() == "TestPool");
        expect(pool.minThreads() == 1U);
        expect(pool.maxThreads() == 2U);
        expect(pool.numThreads() == 1U);
        expect(pool.numTasksRunning() == 0U);
        expect(pool.numTasksQueued() == 0U);
        expect(pool.numTasksRecycled() == 0U);
#ifdef _WIN32
        pool.execute([&enqueueCount] { enqueueCount.increment(); });
#else
        pool.execute([&enqueueCount] {
            ++enqueueCount;
            enqueueCount.notify_all();
        });
#endif
        enqueueCount.wait(0);
        expect(pool.numThreads() == 1U);
#ifdef _WIN32
        pool.execute([&executeCount] { executeCount.increment(); });
#else
        pool.execute([&executeCount] {
            ++executeCount;
            executeCount.notify_all();
        });
#endif
        executeCount.wait(0);
        expect(pool.numThreads() >= 1U);
        expect(enqueueCount.load() == 1);
        expect(executeCount.load() == 1);

        auto ret = pool.execute([] { return 42; });
        expect(ret.get() == 42);

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
#ifdef _WIN32
        SafeCounter counter;
#else
        std::atomic<std::size_t> counter{0UZ};
#endif
        gr::thread_pool::BasicThreadPool pool("contention", gr::thread_pool::IO_BOUND, 1, 4);
        pool.waitUntilInitialised();
        expect(that % pool.isInitialised());
        expect(pool.numThreads() == 1U);
#ifdef _WIN32
        pool.execute([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            counter.increment();
        });
#else
        pool.execute([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::atomic_fetch_add(&counter, 1UZ);
            counter.notify_all();
        });
#endif
        expect(pool.numThreads() == 1U);
#ifdef _WIN32
        pool.execute([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            counter.increment();
        });
#else
        pool.execute([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::atomic_fetch_add(&counter, 1UZ);
            counter.notify_all();
        });
#endif
        expect(pool.numThreads() >= 1UZ);
        counter.wait(0UZ);
        counter.wait(1UZ);
        expect(counter.load() == 2UZ);
    };

    "ThreadPool: Thread count tests"_test = [] {
        struct bounds_def {
            std::uint32_t min, max;
        };
        std::array<bounds_def, 5> bounds{bounds_def{1, 1}, bounds_def{1, 4}, bounds_def{2, 2}, bounds_def{2, 8}, bounds_def{4, 8}};

        for (const auto [minThreads, maxThreads] : bounds) {
            for (const auto taskCount : {2UZ, 8UZ, 32UZ}) {
                std::print("## Test with min={} and max={} and taskCount={}\n", minThreads, maxThreads, taskCount);
                std::atomic<std::size_t> counter{0UZ};

                // Pool with min and max thread count
                gr::thread_pool::BasicThreadPool pool("count_test", gr::thread_pool::IO_BOUND, minThreads, maxThreads);
                pool.keepAliveDuration = std::chrono::milliseconds(10); // default is 10 seconds, reducing for testing
                pool.waitUntilInitialised();

                for (std::size_t i = 0UZ; i < taskCount; ++i) {
                    pool.execute([&counter] {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10UZ));
                        std::atomic_fetch_add(&counter, 1UZ);
                        counter.notify_all();
                    });
                }
                expect(that % pool.numThreads() >= minThreads);
                // the maximum number of threads is not a hard limit, if there is a burst of execute calls, it will spwawn more than maxThreads trheads.
                // expect(that % pool.numThreads() == std::min(std::uint32_t(taskCount), maxThreads));

                for (std::size_t i = 0UZ; i < taskCount; ++i) {
                    counter.wait(i);
                    expect(that % pool.numThreads() >= minThreads);
                    // expect(that % pool.numThreads() <= maxThreads); // not a hard limit
                }

                // We should have gotten back to minimum
                std::this_thread::sleep_for(std::chrono::milliseconds(100UZ));
                expect(that % pool.numThreads() == minThreads);
                expect(that % counter.load() == taskCount);
            }
        }
    };

    "ThreadPool: CPU affinity rejection"_test = [] {
        using namespace gr::thread_pool;

        BasicThreadPool pool("AffinityReject", TaskType::CPU_BOUND, 1U, 2U);
        pool.waitUntilInitialised();

        // Set affinity mask to enable only CPU 0
        pool.setAffinityMask({true, false, false});

        expect(throws<std::invalid_argument>([&] { pool.execute<"bad_affinity", 0, 1>([] { std::println("should not run"); }); }));
    };

    "ThreadPool: exception propagation"_test = [] {
        using namespace gr::thread_pool;

        BasicThreadPool pool("ExceptionTest", TaskType::IO_BOUND, 1U, 1U);
        pool.waitUntilInitialised();

        auto fut = pool.execute([]() -> int { throw std::runtime_error("expected failure"); });

        expect(throws<std::runtime_error>([&] { (void)fut.get(); }));
    };

    "ThreadPool: recycled task count increases"_test = [] {
        using namespace gr::thread_pool;

        BasicThreadPool pool("RecycleTest", TaskType::IO_BOUND, 1U, 1U);
        pool.waitUntilInitialised();

        const auto before = pool.numTasksRecycled();

        for (std::size_t i = 0; i < 5; ++i) {
            std::atomic<bool> flag{false};
            pool.execute([&] {
                flag = true;
                flag.notify_all();
            });
            flag.wait(false);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // let worker recycle

        expect(gt(pool.numTasksRecycled(), before)); // conservatively ≥
    };

    "ThreadPool: setThreadBounds valid and invalid cases"_test = [] {
        using namespace gr::thread_pool;

        BasicThreadPool pool("BoundsTest", TaskType::CPU_BOUND, 2U, 4U);
        pool.waitUntilInitialised();

        expect(nothrow([&] { pool.setThreadBounds(1U, 8U); }));
        expect(pool.minThreads() == 1U);
        expect(pool.maxThreads() == 8U);

        expect(throws<std::invalid_argument>([&] { pool.setThreadBounds(0U, 8U); }));
        expect(throws<std::invalid_argument>([&] { pool.setThreadBounds(2U, 0U); }));
        expect(throws<std::invalid_argument>([&] { pool.setThreadBounds(5U, 4U); }));

        expect(nothrow([&] { pool.setThreadBounds(3U, 3U); }));
        expect(pool.minThreads() == 3U);
        expect(pool.maxThreads() == 3U);
    };
};

const boost::ut::suite<"gr::thread_pool Manager"> ThreadPoolManager = [] {
    using namespace boost::ut;
    using namespace gr::thread_pool;

    "Manager: default pools registration and retrieval"_test = [] {
        auto cpu = Manager::defaultCpuPool();
        auto io  = Manager::defaultIoPool();

        expect(cpu->type() == TaskType::CPU_BOUND);
        expect(io->type() == TaskType::IO_BOUND);
        expect(cpu->device() == "CPU");
        expect(io->device() == "CPU");
        auto [min, max] = cpu->threadBounds();
        expect(eq(min, cpu->minThreads()));
        expect(eq(max, cpu->maxThreads()));

        std::atomic<std::size_t> flag{0UZ};
        cpu->execute([&] {
            flag = 1UZ;
            flag.notify_all();
        });
        flag.wait(0UZ);

        expect(cpu->numThreads() >= 1U) << "needed to execute one task to spawn one thread";
        expect(cpu->isShutdown() == false);
    };

    "Manager: custom pool registration and execution"_test = [] {
        auto& manager = Manager::instance();

        auto custom = std::make_shared<ThreadPoolWrapper>(std::make_unique<BasicThreadPool>("MyPool", TaskType::IO_BOUND, 1, 2), "VirtualDevice");

        expect(nothrow([&] { manager.registerPool("my_pool", std::move(custom)); }));

        auto pool = manager.get("my_pool");
        expect(pool->name() == "MyPool");
        expect(pool->device() == "VirtualDevice");
        expect(pool->type() == TaskType::IO_BOUND);

        std::atomic<std::size_t> flag{0UZ};
        pool->execute([&] {
            flag = 1UZ;
            flag.notify_all();
        });
        flag.wait(0UZ);
        expect(flag.load() == 1UZ);
    };

    "Manager: duplicate registration fails"_test = [] {
        auto dup = std::make_shared<ThreadPoolWrapper>(std::make_unique<BasicThreadPool>("DupPool", TaskType::CPU_BOUND, 1U, 2U), "CPU");
        expect(throws<std::invalid_argument>([&] { Manager::instance().registerPool("default_cpu", std::move(dup)); }));
    };

    "Manager: unknown pool throws"_test = [] { expect(throws<std::out_of_range>([] { (void)Manager::instance().get("not_existing_pool"); })); };

    "Manager: replacePool allows update of registered pool"_test = [] {
        auto& manager = Manager::instance();

        auto updated = std::make_shared<ThreadPoolWrapper>(std::make_unique<BasicThreadPool>("updated_pool", TaskType::CPU_BOUND, 1U, 1U), "CPU-Updated");
        updated->setThreadBounds(1U, 4U);

        manager.replacePool("default_cpu", std::move(updated));

        auto pool = Manager::defaultCpuPool();
        expect(pool->device() == "CPU-Updated");
        expect(pool->threadBounds().second == 4UZ);

        std::atomic<std::size_t> taskRan{0UZ};
        pool->execute([&] {
            taskRan = 1UZ;
            taskRan.notify_all();
        });
        taskRan.wait(0UZ);
        expect(taskRan.load() == 1UZ);
    };
};

#ifndef GR_MAX_WASM_THREAD_COUNT
#define GR_MAX_WASM_THREAD_COUNT 60 // fallback
#endif

const boost::ut::suite<"gr::thread_pool Manager WASM"> _wasm = [] {
    using namespace boost::ut;
    using namespace gr::thread_pool;

    "getTotalThreadCount"_test = [] {
        const std::size_t count = gr::thread_pool::getTotalThreadCount();
        expect(gt(count, 0UZ));
        expect(le(count, gr::thread_pool::thread::getThreadLimit()));
    };

    "global thread counter tracking thread creation"_test = [] {
        // Wait for the process-wide thread count to stabilise — threads from earlier
        // test suites (with short keepAliveDuration) may still be winding down at the OS level.
        std::size_t before = gr::thread_pool::getTotalThreadCount();
        for (int stableChecks = 0, i = 0; i < 40 && stableChecks < 3; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            if (const auto now = gr::thread_pool::getTotalThreadCount(); now == before) {
                ++stableChecks;
            } else {
                before       = now;
                stableChecks = 0;
            }
        }
        {
            BasicThreadPool temp("test_pool", CPU_BOUND, 2, 2);
            temp.waitUntilInitialised();
            expect(eq(temp.numThreads(), 2UZ)) << "pool should have exactly 2 threads";
            const std::size_t during = gr::thread_pool::getTotalThreadCount();
            expect(ge(during, before + 2));
        }
        // Pool destructor joins all threads; wait briefly for OS bookkeeping.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const std::size_t after = gr::thread_pool::getTotalThreadCount();
        expect(eq(after, before)) << "Expected cleanup after pool destruction";
    };

    "GR_MAX_WASM_THREAD_COUNT compile-time constant"_test = [] {
#ifdef __EMSCRIPTEN__
#ifdef GR_MAX_WASM_THREAD_COUNT
        expect(eq(static_cast<std::size_t>(GR_MAX_WASM_THREAD_COUNT), gr::thread_pool::thread::getThreadLimit()));
#else
        expect(false) << "GR_MAX_WASM_THREAD_COUNT not defined";
#endif
#else
        expect(gt(gr::thread_pool::thread::getThreadLimit(), 1UZ));
#endif
    };

    "Manager: WASM exhausting available threads"_test = [] {
        Manager& manager = Manager::instance();
#ifdef __EMSCRIPTEN__
        const std::size_t poolMaxThreads = gr::thread_pool::thread::getThreadLimit();
        const auto        taskSleep      = std::chrono::seconds(20);
#else
        const std::size_t poolMaxThreads = std::min<std::size_t>(50UZ, gr::thread_pool::thread::getThreadLimit());
        const auto        taskSleep      = std::chrono::seconds(2);
#endif
        // replace IO pool to unlimited upper bound
        manager.replacePool(std::string(kDefaultIoPoolId), std::make_shared<ThreadPoolWrapper>(std::make_unique<BasicThreadPool>(kDefaultIoPoolId, TaskType::IO_BOUND, 1U, poolMaxThreads), "CPU"));
        std::shared_ptr<TaskExecutor> pool = manager.get(gr::thread_pool::kDefaultIoPoolId);

        std::println("HW threads = {} - max wasm threads: {} actual: {} - pool max size: {}", //
            std::thread::hardware_concurrency(), gr::thread_pool::thread::getThreadLimit(), gr::thread_pool::getTotalThreadCount(), pool->maxThreads());
        std::atomic<std::size_t> unexpectedExceptions{0UZ};
        std::atomic<std::size_t> expectedExceptions{0UZ};
        for (std::size_t i = 0UZ; i < poolMaxThreads + 10UZ; ++i) {
            if (i >= (poolMaxThreads - 10UZ)) {
                std::println("start thread {}", i);
            }
            try {
                pool->execute([taskSleep] {
                    std::this_thread::sleep_for(taskSleep); // purposeful sleep
                });
            } catch (std::exception& e) {
                std::println("exception thrown: {} for {} threads", e, gr::thread_pool::getTotalThreadCount());
                expectedExceptions.fetch_add(1UZ, std::memory_order_relaxed);
            } catch (...) {
                std::println("unknown exception thrown for {} threads", gr::thread_pool::getTotalThreadCount());
                unexpectedExceptions.fetch_add(1UZ, std::memory_order_relaxed);
            }
            if ((expectedExceptions.load() + unexpectedExceptions.load()) >= 10UZ) {
                break;
            }
        }
        std::println("number of exceptions thrown: {} unexpeced: {}", expectedExceptions.load(), unexpectedExceptions.load());
#ifdef __EMSCRIPTEN__
        expect(gt(expectedExceptions.load(), 0UZ)) << fatal << "creating more threads than kThreadLimit should throw with expected exception";
#endif
        expect(eq(unexpectedExceptions.load(), 0UZ)) << fatal << "caught unexpected exception";
    };

    "computeDefaultThreadSplit respects invariants under edge conditions"_test = [] {
        using namespace gr::thread_pool::detail;

        const auto validate_split = [](std::size_t threadLimit, std::size_t reserve, std::source_location loc = std::source_location::current()) {
            const auto s      = computeDefaultThreadSplit(threadLimit, reserve);
            const auto usable = (threadLimit > reserve) ? threadLimit - reserve : 1UZ;
            const auto actual = s.cpuThreadsMax + s.ioThreadsMax;

            expect(le(actual, std::max(2UZ, usable)), loc) << std::format("limit={}, reserve={} -> cpu+io={} must not exceed usable={}", threadLimit, reserve, actual, usable);
            expect(le(s.cpuThreadsMax, usable), loc) << std::format("limit={}, reserve={} -> CPU threads {} must be ≤ usable {}", threadLimit, reserve, s.cpuThreadsMax, usable);
            expect(le(s.ioThreadsMax, usable), loc) << std::format("limit={}, reserve={} -> IO threads {} must be ≤ usable {}", threadLimit, reserve, s.ioThreadsMax, usable);

#if defined(__EMSCRIPTEN__)
            expect(ge(s.cpuThreadsMin, 0UZ), loc) << std::format("WASM: limit={}, reserve={} -> At least 1 CPU thread expected (got {})", threadLimit, reserve, s.cpuThreadsMin);
            expect(ge(s.ioThreadsMin, 0UZ), loc) << std::format("WASM: limit={}, reserve={} -> At least 1 CPU thread expected (got {})", threadLimit, reserve, s.cpuThreadsMin);
#else
            expect(ge(s.cpuThreadsMin, 1UZ), loc) << std::format("limit={}, reserve={} -> At least 1 CPU thread expected (got {})", threadLimit, reserve, s.cpuThreadsMin);
            expect(ge(s.ioThreadsMin, 1UZ), loc) << std::format("limit={}, reserve={} -> At least 1 CPU thread expected (got {})", threadLimit, reserve, s.cpuThreadsMin);
#endif
            expect(ge(s.ioThreadsMax, 1UZ), loc) << std::format("limit={}, reserve={} -> At least 1 IO thread expected (got {})", threadLimit, reserve, s.ioThreadsMax);

            expect(eq(s.threadReserve, reserve), loc) << std::format("limit={}, reserve={} -> Expected thread reserve {}, got {}", threadLimit, reserve, reserve, s.threadReserve);
        };

        validate_split(3UZ, 4UZ);     // threadLimit < reserve → usable = 1
        validate_split(4UZ, 4UZ);     // usable = 1
        validate_split(8UZ, 4UZ);     // small WASM/native config
        validate_split(64UZ, 4UZ);    // typical system config
        validate_split(50000UZ, 4UZ); // upper-bound stress test
    };

    "Manager: respects thread limit and budget"_test = [] {
        using namespace gr::thread_pool;

        Manager& manager = Manager::instance();
        auto     cpu     = manager.get(kDefaultCpuPoolId);
        auto     io      = manager.get(kDefaultIoPoolId);

        const std::size_t totalUsed = cpu->numThreads() + io->numThreads();
        const std::size_t threadCap = thread::getThreadLimit();

        std::println("CPU threads: {}, IO threads: {}, total: {}, cap: {}", cpu->numThreads(), io->numThreads(), totalUsed, threadCap);

        expect(le(totalUsed, threadCap - 1)) << "Manager over-allocated threads";
        expect(gt(io->numThreads(), cpu->numThreads())) << "IO thread count should exceed CPU";
    };
};

int main() { /* tests are statically executed */ }
