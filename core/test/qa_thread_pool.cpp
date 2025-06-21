#include <boost/ut.hpp>

#include <gnuradio-4.0/thread/thread_pool.hpp>

const boost::ut::suite<"gr::thread_pool GR4 default"> defaultThreadPool = [] {
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
        expect(pool.minThreads() == 1U);
        expect(pool.maxThreads() == 2U);
        expect(pool.numThreads() == 1U);
        expect(pool.numTasksRunning() == 0U);
        expect(pool.numTasksQueued() == 0U);
        expect(pool.numTasksRecycled() == 0U);
        pool.execute([&enqueueCount] {
            ++enqueueCount;
            enqueueCount.notify_all();
        });
        enqueueCount.wait(0);
        expect(pool.numThreads() == 1U);
        pool.execute([&executeCount] {
            ++executeCount;
            executeCount.notify_all();
        });
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
        std::atomic<std::size_t>         counter{0UZ};
        gr::thread_pool::BasicThreadPool pool("contention", gr::thread_pool::IO_BOUND, 1, 4);
        pool.waitUntilInitialised();
        expect(that % pool.isInitialised());
        expect(pool.numThreads() == 1U);
        pool.execute([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::atomic_fetch_add(&counter, 1UZ);
            counter.notify_all();
        });
        expect(pool.numThreads() == 1U);
        pool.execute([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::atomic_fetch_add(&counter, 1UZ);
            counter.notify_all();
        });
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

int main() { /* tests are statically executed */ }
