#include <boost/ut.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include <gnuradio-4.0/device/ParallelFor.hpp>
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

const suite<"device::SchedulerRegistry + parallelFor"> _registryAndLaunch = [] {
    "tryResolve reports absence instead of falling back to the CPU"_test = [] {
        auto& registry = gr::device::SchedulerRegistry::instance();
        expect(registry.tryResolve("fpga:absent:7") == nullptr);
        expect(registry.resolve("fpga:absent:7").backend() == gr::device::DeviceBackend::CPU_Fallback) << "resolve() keeps its silent CPU fallback";
    };

    "tryResolve matches a registered context through a shorter prefix"_test = [] {
        auto& registry = gr::device::SchedulerRegistry::instance();
        registry.registerContext("tpu:mock", std::make_unique<gr::device::DeviceContextCpu>());

        auto* scheduler = registry.tryResolve("tpu:mock:3");
        expect(scheduler != nullptr);
        expect(scheduler->backend() == gr::device::DeviceBackend::CPU_Fallback);
    };

    "parallelFor runs the kernel on a context without a device backend"_test = [] {
        gr::device::DeviceContextCpu context;
        std::array<int, 4>           values{};

        gr::device::parallelFor(context, values.size(), [data = values.data()](std::size_t i) { data[i] = static_cast<int>(2UZ * i); });

        expect(eq(values[0], 0));
        expect(eq(values[3], 6));
    };
};

int main() { /* not needed for UT */ }
