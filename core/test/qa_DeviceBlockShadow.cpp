#include <boost/ut.hpp>

#include <cstddef>
#include <string>
#include <utility>

#include <gnuradio-4.0/device/DeviceBlockShadow.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::test {

/// a context that answers like the CPU backing but reports what the shadow asked it for; `DeviceContextCpu` is
/// `final`, so the counting happens beside it rather than under it
struct CountingContext final : gr::device::DeviceContext {
    gr::device::DeviceContextCpu backing{};
    std::size_t                  allocations   = 0UZ;
    std::size_t                  deallocations = 0UZ;
    std::size_t                  waits         = 0UZ;

    [[nodiscard]] gr::device::DeviceBackend backend() const noexcept override { return backing.backend(); }
    [[nodiscard]] gr::device::DeviceType    deviceType() const noexcept override { return backing.deviceType(); }
    [[nodiscard]] std::string               shortName() const override { return backing.shortName(); }
    [[nodiscard]] std::string               name() const override { return backing.name(); }
    [[nodiscard]] std::string               version() const override { return backing.version(); }

    void wait() override { ++waits; }

    [[nodiscard]] gr::device::DeviceBuffer allocate(std::size_t bytes, std::size_t align, gr::device::Residency wanted) override {
        ++allocations;
        return backing.allocate(bytes, align, wanted);
    }
    void deallocate(gr::device::DeviceBuffer buffer) override {
        ++deallocations;
        backing.deallocate(buffer);
    }
    void upload(const void* host, gr::device::DeviceBuffer destination, std::size_t bytes) override { backing.upload(host, destination, bytes); }
    void download(gr::device::DeviceBuffer source, void* host, std::size_t bytes) override { backing.download(source, host, bytes); }

    [[nodiscard]] std::size_t liveAllocations() const noexcept { return allocations - deallocations; }
};

} // namespace gr::test

const boost::ut::suite<"DeviceBlockShadow"> deviceBlockShadowTests = [] {
    using namespace boost::ut;
    using gr::device::DeviceBlockShadow;
    using gr::test::CountingContext;

    "a fresh shadow owns no memory and names no context"_test = [] {
        DeviceBlockShadow shadow;
        expect(!shadow.mirror);
        expect(!shadow.control);
        expect(eq(shadow.controlBytes, 0UZ));
        expect(shadow.context == nullptr);
        expect(eq(shadow.epoch, DeviceBlockShadow::kNeverRefreshed));
        expect(!shadow.workInFlight);
    };

    "a mirror acquired twice on one context is the same allocation"_test = [] {
        CountingContext   context;
        DeviceBlockShadow shadow;

        const gr::device::DeviceBuffer first  = shadow.acquire(context, 256UZ, alignof(std::max_align_t));
        const gr::device::DeviceBuffer second = shadow.acquire(context, 256UZ, alignof(std::max_align_t));

        expect(first.token != 0UZ);
        expect(eq(first.token, second.token)) << "a second dispatch on the same context must reuse the mirror, not rebuild it";
        expect(eq(context.allocations, 1UZ));
    };

    "a mirror acquired on a different context is reallocated there, and the settings epoch is invalidated"_test = [] {
        CountingContext   first;
        CountingContext   second;
        DeviceBlockShadow shadow;

        std::ignore  = shadow.acquire(first, 256UZ, alignof(std::max_align_t));
        shadow.epoch = 7UZ;
        std::ignore  = shadow.acquire(second, 256UZ, alignof(std::max_align_t));

        expect(shadow.context == &second);
        expect(eq(first.liveAllocations(), 0UZ)) << "the first context must free what it allocated, or the mirror leaks on every context switch";
        expect(eq(second.allocations, 1UZ));
        expect(eq(shadow.epoch, DeviceBlockShadow::kNeverRefreshed)) << "memory acquired elsewhere holds no settings, so the next dispatch must re-seat it";
    };

    "a control area grows on demand and is never handed back smaller"_test = [] {
        CountingContext   context;
        DeviceBlockShadow shadow;
        std::ignore = shadow.acquire(context, 256UZ, alignof(std::max_align_t));

        std::byte* wide   = shadow.controlArea(context, 4096UZ);
        std::byte* narrow = shadow.controlArea(context, 16UZ);

        expect(wide != nullptr);
        expect(wide == narrow) << "shrinking would free accounting a deferred kernel is still writing";
        expect(ge(shadow.controlBytes, 4096UZ));

        std::byte* grown = shadow.controlArea(context, 8192UZ);
        expect(grown != nullptr);
        expect(ge(shadow.controlBytes, 8192UZ));
    };

    "a control area asked for on a new context is freed by the one that allocated it"_test = [] {
        CountingContext   first;
        CountingContext   second;
        DeviceBlockShadow shadow;

        std::ignore = shadow.acquire(first, 256UZ, alignof(std::max_align_t));
        std::ignore = shadow.controlArea(first, 1024UZ);
        std::ignore = shadow.controlArea(second, 1024UZ);

        expect(shadow.context == &second);
        expect(eq(first.liveAllocations(), 0UZ)) << "freeing a buffer through a context that did not allocate it is undefined on a real backend";
        expect(eq(second.allocations, 1UZ));

        const gr::device::DeviceBuffer mirror = shadow.acquire(second, 256UZ, alignof(std::max_align_t));
        expect(static_cast<bool>(mirror)) << "the mirror went with the old context, so the next dispatch must get a new one rather than an invalid buffer";
        expect(mirror.devicePointer<std::byte>() != nullptr);
        expect(eq(second.allocations, 2UZ));
    };

    "a control area asked for before any mirror is still owned, and released"_test = [] {
        CountingContext   context;
        DeviceBlockShadow shadow;

        std::ignore = shadow.controlArea(context, 1024UZ);
        expect(eq(context.allocations, 1UZ));

        shadow.release();
        expect(eq(context.liveAllocations(), 0UZ)) << "a shadow that never acquired a mirror must still free its control area";
    };

    "releasing twice is harmless and leaves nothing behind"_test = [] {
        CountingContext   context;
        DeviceBlockShadow shadow;
        std::ignore = shadow.acquire(context, 256UZ, alignof(std::max_align_t));
        std::ignore = shadow.controlArea(context, 1024UZ);

        shadow.release();
        shadow.release();

        expect(eq(context.liveAllocations(), 0UZ));
        expect(eq(context.deallocations, 2UZ)) << "each of the two buffers is freed exactly once";
        expect(shadow.context == nullptr);
        expect(eq(shadow.controlBytes, 0UZ));
    };

    "a moved-from shadow owns nothing, so each buffer is freed once"_test = [] {
        CountingContext context;
        {
            DeviceBlockShadow source;
            std::ignore = source.acquire(context, 256UZ, alignof(std::max_align_t));
            std::ignore = source.controlArea(context, 1024UZ);

            DeviceBlockShadow moved{std::move(source)};
            expect(static_cast<bool>(moved.mirror));
            expect(static_cast<bool>(moved.control));
            expect(eq(moved.controlBytes, 1024UZ));
            expect(!source.mirror);               // NOLINT(bugprone-use-after-move) — the moved-from state is the assertion
            expect(!source.control);              // NOLINT(bugprone-use-after-move)
            expect(eq(source.controlBytes, 0UZ)); // NOLINT(bugprone-use-after-move)
        }
        expect(eq(context.liveAllocations(), 0UZ));
        expect(eq(context.deallocations, 2UZ)) << "a double free here is what a copied buffer token would look like";
    };

    "work left in flight is awaited before the memory a kernel reads is touched"_test = [] {
        CountingContext   context;
        DeviceBlockShadow shadow;
        std::ignore         = shadow.acquire(context, 256UZ, alignof(std::max_align_t));
        shadow.workInFlight = true;

        std::ignore = shadow.controlArea(context, 4096UZ);
        expect(eq(context.waits, 1UZ)) << "growing the control area reallocates it, which a kernel still writing it must not see";
        expect(!shadow.workInFlight);

        shadow.workInFlight = true;
        shadow.release();
        expect(eq(context.waits, 2UZ));
    };

    "awaiting without a context is a no-op rather than a null dereference"_test = [] {
        DeviceBlockShadow shadow;
        shadow.workInFlight = true;
        shadow.awaitWorkInFlight();
        expect(shadow.workInFlight) << "with no context there is nothing to wait on, so the flag stands";
    };
};

int main() { return 0; }
