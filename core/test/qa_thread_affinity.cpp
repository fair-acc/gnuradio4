#include <boost/ut.hpp>

#include <fmt/format.h>

#include <gnuradio-4.0/thread/thread_affinity.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

const boost::ut::suite ThreadAffinityTests = [] {
    using namespace boost::ut;

    "thread_exception"_test = [] {
        expect(nothrow([]{gr::thread_pool::thread::thread_exception();}));
        expect(gr::thread_pool::thread::thread_exception().name() == "thread_exception"_b);
        expect(gr::thread_pool::thread::thread_exception().message(-1) == "unknown threading error code -1"_b);
        expect(gr::thread_pool::thread::thread_exception().message(-2) == "unknown threading error code -2"_b);
        expect(!gr::thread_pool::thread::thread_exception().message(gr::thread_pool::thread::THREAD_UNINITIALISED).starts_with("unknown threading error code"));
        expect(!gr::thread_pool::thread::thread_exception().message(gr::thread_pool::thread::THREAD_ERROR_UNKNOWN).starts_with("unknown threading error code"));
        expect(!gr::thread_pool::thread::thread_exception().message(gr::thread_pool::thread::THREAD_VALUE_RANGE).starts_with("unknown threading error code"));
        expect(!gr::thread_pool::thread::thread_exception().message(gr::thread_pool::thread::THREAD_ERANGE).starts_with("unknown threading error code"));
    };

    "thread_helper"_test = [] {
#if not defined(__EMSCRIPTEN__)
        expect(that % gr::thread_pool::thread::detail::getEnumPolicy(SCHED_FIFO) == gr::thread_pool::thread::Policy::FIFO);
        expect(that % gr::thread_pool::thread::detail::getEnumPolicy(SCHED_RR) == gr::thread_pool::thread::Policy::ROUND_ROBIN);
        expect(that % gr::thread_pool::thread::detail::getEnumPolicy(SCHED_OTHER) == gr::thread_pool::thread::Policy::OTHER);
#endif
        expect(that % gr::thread_pool::thread::detail::getEnumPolicy(-1) == gr::thread_pool::thread::Policy::UNKNOWN);
        expect(that % gr::thread_pool::thread::detail::getEnumPolicy(-2) == gr::thread_pool::thread::Policy::UNKNOWN);
    };

#if not defined(__EMSCRIPTEN__)
    "basic thread affinity"_test = [] {
        using namespace gr::thread_pool;
        std::atomic<bool>    run         = true;
        const auto           dummyAction = [&run]() { while (run) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); } };
        std::thread          testThread(dummyAction);

        constexpr std::array threadMap = { true, false, false, false };
        thread::setThreadAffinity(threadMap, testThread);
        auto affinity = thread::getThreadAffinity(testThread);
        bool equal    = true;
        for (size_t i = 0; i < std::min(threadMap.size(), affinity.size()); i++) {
            if (threadMap[i] != affinity[i]) {
                equal = false;
            }
        }
        expect(equal) << fmt::format("set {{{}}} affinity map does not match get {{{}}} map", fmt::join(threadMap, ", "), fmt::join(affinity, ", "));

        // tests w/o thread argument
        constexpr std::array threadMapOn = { true, true };
        thread::setThreadAffinity(threadMapOn);
        affinity = thread::getThreadAffinity();
        for (size_t i = 0; i < std::min(threadMapOn.size(), affinity.size()); i++) {
            if (threadMapOn[i] != affinity[i]) {
                equal = false;
            }
        }
        expect(equal) << fmt::format("set {{{}}} affinity map does not match get {{{}}} map", fmt::join(threadMap, ", "), fmt::join(affinity, ", "));

        std::thread bogusThread;
        expect(throws<std::system_error>([&]{ thread::getThreadAffinity(bogusThread); }));
        expect(throws<std::system_error>([&]{ thread::setThreadAffinity(threadMapOn, bogusThread); }));

        run = false;
        testThread.join();
    };

    "basic process affinity"_test = [] {
        using namespace gr::thread_pool;
        constexpr std::array threadMap = { true, false, false, false };
        thread::setProcessAffinity(threadMap);
        auto affinity = thread::getProcessAffinity();
        bool equal    = true;
        for (size_t i = 0; i < std::min(threadMap.size(), affinity.size()); i++) {
            if (threadMap[i] != affinity[i]) {
                equal = false;
            }
        }
        expect(equal) << fmt::format("set {{{}}} affinity map does not match get {{{}}} map", fmt::join(threadMap, ", "), fmt::join(affinity, ", "));
        constexpr std::array threadMapOn = { true, true, true, true };
        thread::setProcessAffinity(threadMapOn);
        expect(throws<std::system_error>([&]{ thread::getProcessAffinity(-1); }));
        expect(throws<std::system_error>([&]{ thread::setProcessAffinity(threadMapOn, -1); }));
    };

    "ThreadName"_test = [] {
        using namespace gr::thread_pool;
        expect(!thread::getThreadName().empty()) << "Thread name shouldn't be empty";
        expect(nothrow([]{ thread::setThreadName("testCoreName"); }));
        expect(thread::getThreadName() == "testCoreName"_b);

        std::atomic<bool> run         = true;
        const auto        dummyAction = [&run]() { while (run) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); } };
        std::thread       testThread(dummyAction);
        expect(!thread::getThreadName(testThread).empty()) << "Thread Name shouldn't be empty";
        expect(nothrow([&]{ thread::setThreadName("testThreadName", testThread); }));
        thread::setThreadName("testThreadName", testThread);
        expect(thread::getThreadName(testThread) == "testThreadName"_b);

        std::thread uninitialisedTestThread;
        expect(throws<std::system_error>([&]{ thread::getThreadName(uninitialisedTestThread); }));
        expect(throws<std::system_error>([&]{ thread::setThreadName("name", uninitialisedTestThread); }));
        run = false;
        testThread.join();
    };

    "ProcessName"_test = [] {
        using namespace gr::thread_pool;
        expect(!thread::getProcessName().empty()) << "Process name shouldn't be empty";
        expect(that % thread::getProcessName() == thread::getProcessName(thread::detail::getPid()));

        expect(nothrow([]{ thread::setProcessName("TestProcessName"); }));
        expect(thread::getProcessName() == "TestProcessName"_b);
    };

    "ProcessSchedulingParameter"_test = [] {
        using namespace gr::thread_pool::thread;
        struct SchedulingParameter param = getProcessSchedulingParameter();
        expect(that % param.policy == OTHER);
        expect(that % param.priority == 0);

        expect(nothrow([]{ setProcessSchedulingParameter(OTHER, 0); }));
        expect(throws<std::system_error>([]{ setProcessSchedulingParameter(OTHER, 0, -1); }));
        expect(throws<std::system_error>([]{ setProcessSchedulingParameter(OTHER, 4); }));
        expect(throws<std::system_error>([]{ setProcessSchedulingParameter(ROUND_ROBIN, 5); })); // missing rights -- because most users do not have CAP_SYS_NICE rights by default -- hard to unit-test
        param = getProcessSchedulingParameter();
        expect(that % param.policy == OTHER);
        expect(that % param.priority == 0);

        expect(throws<std::system_error>([]{ getProcessSchedulingParameter(-1); }));
        expect(throws<std::system_error>([]{ setProcessSchedulingParameter(ROUND_ROBIN, 5, -1); }));

        expect(that % gr::thread_pool::thread::detail::getEnumPolicy(SCHED_FIFO) == gr::thread_pool::thread::FIFO);
        expect(that % gr::thread_pool::thread::detail::getEnumPolicy(SCHED_RR) == gr::thread_pool::thread::ROUND_ROBIN);
        expect(that % gr::thread_pool::thread::detail::getEnumPolicy(SCHED_OTHER) == gr::thread_pool::thread::OTHER);
    };

    "ThreadSchedulingParameter"_test = [] {
        std::atomic<bool>     run         = true;
        const auto            dummyAction = [&run]() { while (run) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); } };
        std::thread           testThread(dummyAction);
        std::thread           bogusThread;

        using namespace gr::thread_pool::thread;
        struct SchedulingParameter param = getThreadSchedulingParameter(testThread);
        expect(that % param.policy == OTHER);
        expect(that % param.priority == 0);

        setThreadSchedulingParameter(OTHER, 0, testThread);
        setThreadSchedulingParameter(OTHER, 0);
        expect(throws<std::system_error>([&]{ setThreadSchedulingParameter(OTHER, 0, bogusThread); }));
        expect(throws<std::system_error>([&]{ setThreadSchedulingParameter(OTHER, 4, testThread); }));
        expect(throws<std::system_error>([&]{ setThreadSchedulingParameter(OTHER, 4); }));
        expect(throws<std::system_error>([&]{ setThreadSchedulingParameter(ROUND_ROBIN, 5, testThread); })); // missing rights -- because most users do not have CAP_SYS_NICE rights by default -- hard to unit-test
        expect(throws<std::system_error>([&]{ setThreadSchedulingParameter(ROUND_ROBIN, 5); }));             // missing rights -- because most users do not have CAP_SYS_NICE rights by default -- hard to unit-test
        param = getThreadSchedulingParameter(testThread);
        expect(that % param.policy == OTHER);

        expect(throws<std::system_error>([&]{ getThreadSchedulingParameter(bogusThread); }));
        expect(throws<std::system_error>([&]{ setThreadSchedulingParameter(ROUND_ROBIN, 5, bogusThread); }));

        run = false;
        testThread.join();
    };
#endif
};

int
main() { /* tests are statically executed */
}
