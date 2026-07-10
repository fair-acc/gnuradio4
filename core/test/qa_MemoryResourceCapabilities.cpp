#include <boost/ut.hpp>

#include <gnuradio-4.0/MemoryAllocators.hpp>

#include <memory_resource>
#include <thread>
#include <vector>

namespace {
struct DummyResource final : std::pmr::memory_resource {
    void* do_allocate(std::size_t bytes, std::size_t alignment) override { return ::operator new(bytes, std::align_val_t{alignment}); }
    void  do_deallocate(void* p, std::size_t, std::size_t alignment) override { ::operator delete(p, std::align_val_t{alignment}); }
    bool  do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
};
} // namespace

const boost::ut::suite<"MemoryResourceCapabilities"> _capabilities = [] {
    using namespace boost::ut;

    "an unregistered resource reads back as ordinary host memory"_test = [] {
        DummyResource resource;
        expect(!gr::usesMMAP(&resource));
        expect(!gr::isDeviceOnly(&resource));
    };

    "a null resource is safe to query"_test = [] {
        expect(!gr::usesMMAP(nullptr));
        expect(!gr::isDeviceOnly(nullptr));
    };

    "the default-resource and new_delete_resource are never mmap or device-only"_test = [] {
        expect(!gr::usesMMAP(std::pmr::new_delete_resource()));
        expect(!gr::isDeviceOnly(std::pmr::new_delete_resource()));
        expect(!gr::usesMMAP(std::pmr::get_default_resource()));
    };

    "declared capabilities are returned verbatim"_test = [] {
        DummyResource host;
        DummyResource device;
        gr::registerMemoryResourceCapabilities(&host, {.usesMMAP = true, .deviceOnly = false});
        gr::registerMemoryResourceCapabilities(&device, {.usesMMAP = true, .deviceOnly = true});

        expect(gr::usesMMAP(&host));
        expect(!gr::isDeviceOnly(&host));
        expect(gr::usesMMAP(&device));
        expect(gr::isDeviceOnly(&device));

        gr::deregisterMemoryResourceCapabilities(&host);
        gr::deregisterMemoryResourceCapabilities(&device);
    };

    "deregistration reverts to the conservative answer"_test = [] {
        DummyResource resource;
        gr::registerMemoryResourceCapabilities(&resource, {.usesMMAP = true, .deviceOnly = true});
        expect(gr::isDeviceOnly(&resource));
        gr::deregisterMemoryResourceCapabilities(&resource);
        expect(!gr::usesMMAP(&resource)) << "a withdrawn resource must not keep claiming mmap";
        expect(!gr::isDeviceOnly(&resource)) << "a withdrawn resource must not keep claiming device-only";
    };

    "deregistering an unknown resource is a no-op"_test = [] {
        DummyResource never;
        expect(nothrow([&never] { gr::deregisterMemoryResourceCapabilities(&never); }));
        expect(nothrow([] { gr::deregisterMemoryResourceCapabilities(nullptr); }));
    };

    "re-declaration replaces rather than duplicates"_test = [] {
        DummyResource resource;
        gr::registerMemoryResourceCapabilities(&resource, {.usesMMAP = true, .deviceOnly = false});
        gr::registerMemoryResourceCapabilities(&resource, {.usesMMAP = true, .deviceOnly = true});
        expect(gr::isDeviceOnly(&resource));
        gr::deregisterMemoryResourceCapabilities(&resource); // one withdrawal must suffice
        expect(!gr::usesMMAP(&resource));
    };

    "the scoped helper declares for exactly its own lifetime"_test = [] {
        DummyResource resource;
        {
            const gr::ScopedMemoryResourceCapabilities declaration{&resource, {.usesMMAP = true, .deviceOnly = true}};
            expect(gr::usesMMAP(&resource));
            expect(gr::isDeviceOnly(&resource));
        }
        expect(!gr::usesMMAP(&resource));
        expect(!gr::isDeviceOnly(&resource));
    };

    "a moved-from scoped helper does not withdraw twice"_test = [] {
        DummyResource resource;
        {
            gr::ScopedMemoryResourceCapabilities declaration{&resource, {.usesMMAP = true, .deviceOnly = true}};
            gr::ScopedMemoryResourceCapabilities moved{std::move(declaration)};
            expect(gr::isDeviceOnly(&resource)) << "the moved-to helper owns the declaration";
        }
        expect(!gr::isDeviceOnly(&resource));
    };

    "many resources are tracked independently past the reserved capacity"_test = [] {
        constexpr std::size_t      kCount = 40UZ; // > the reserve(16) hint, so growth is exercised
        std::vector<DummyResource> resources(kCount);
        for (std::size_t i = 0UZ; i < kCount; ++i) {
            gr::registerMemoryResourceCapabilities(&resources[i], {.usesMMAP = true, .deviceOnly = (i % 2UZ) == 0UZ});
        }
        for (std::size_t i = 0UZ; i < kCount; ++i) {
            expect(gr::usesMMAP(&resources[i]));
            expect(eq(gr::isDeviceOnly(&resources[i]), (i % 2UZ) == 0UZ)) << "entry " << i << " must keep its own flags";
        }
        for (std::size_t i = 0UZ; i < kCount; ++i) {
            gr::deregisterMemoryResourceCapabilities(&resources[i]);
        }
        for (std::size_t i = 0UZ; i < kCount; ++i) {
            expect(!gr::usesMMAP(&resources[i]));
        }
    };

    "concurrent declaration and query stay consistent"_test = [] {
        constexpr std::size_t      kPerThread = 32UZ;
        constexpr std::size_t      kThreads   = 4UZ;
        std::vector<DummyResource> resources(kThreads * kPerThread);
        std::vector<std::thread>   threads;
        for (std::size_t t = 0UZ; t < kThreads; ++t) {
            threads.emplace_back([&resources, t] {
                for (std::size_t i = 0UZ; i < kPerThread; ++i) {
                    std::pmr::memory_resource* resource = &resources[t * kPerThread + i];
                    gr::registerMemoryResourceCapabilities(resource, {.usesMMAP = true, .deviceOnly = true});
                    std::ignore = gr::usesMMAP(resource);
                    gr::deregisterMemoryResourceCapabilities(resource);
                }
            });
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
        for (DummyResource& resource : resources) {
            expect(!gr::usesMMAP(&resource)) << "every declaration was withdrawn";
        }
    };
};

int main() { /* not needed for boost::ut */ }
