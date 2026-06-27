#include <boost/ut.hpp>

#include <atomic>
#include <vector>

#include <gnuradio-4.0/device/SchedulerRegistry.hpp>
#include <gnuradio-4.0/execution/gpu_scheduler.hpp>
#include <gnuradio-4.0/execution/pool_scheduler.hpp>

using namespace boost::ut;
namespace ex = gr::execution;

const suite<"device::DeviceScheduler"> tests =
    [] {
        "schedule + then produces value"_test = [] {
            gr::device::DeviceContextCpu ctx;
            ex::DeviceScheduler          sched(ctx);

            auto r = ex::sync_wait(sched.schedule() | ex::then([] { return 42; }));
            expect(r.has_value());
            expect(eq(std::get<0>(*r), 42));
        };

        "continues_on hops from CPU pool to device scheduler"_test = [] {
            gr::device::DeviceContextCpu ctx;
            ex::DeviceScheduler          deviceSched(ctx);
            auto                         cpuSched = ex::cpuScheduler();

            std::atomic<int> result{0};
            auto             r = ex::sync_wait(cpuSched.schedule() | ex::then([&result] { result.store(1); }) | ex::continues_on(deviceSched) | ex::then([&result] { result.store(result.load() + 10); }));
            expect(r.has_value());
            expect(eq(result.load(), 11));
        };

        "DeviceScheduler reports CPU_Fallback backend"_test = [] {
            gr::device::DeviceContextCpu ctx;
            ex::DeviceScheduler          sched(ctx);
            expect(sched.backend() == gr::device::DeviceBackend::CPU_Fallback);
        };

        "SchedulerRegistry resolves registered context"_test = [] {
            auto ctx = std::make_unique<gr::device::DeviceContextCpu>();
            gr::device::SchedulerRegistry::instance().registerContext("device:test", std::move(ctx));

            auto& sched = gr::device::SchedulerRegistry::instance().resolve("device:test");
            expect(sched.backend() == gr::device::DeviceBackend::CPU_Fallback);
        };

        "SchedulerRegistry falls back for unknown domain"_test = [] {
            auto& sched = gr::device::SchedulerRegistry::instance().resolve("gpu:nonexistent:99");
            expect(sched.backend() == gr::device::DeviceBackend::CPU_Fallback);
        };

        "SchedulerRegistry prefix matching"_test = [] {
            auto ctx = std::make_unique<gr::device::DeviceContextCpu>();
            gr::device::SchedulerRegistry::instance().registerContext("gpu:sycl", std::move(ctx));

            auto& sched = gr::device::SchedulerRegistry::instance().resolve("gpu:sycl:0");
            expect(sched.backend() == gr::device::DeviceBackend::CPU_Fallback);
        };
};

int main() { /* not needed for UT */ }
