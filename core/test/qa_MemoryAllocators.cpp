#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#include <boost/ut.hpp>
#pragma GCC diagnostic pop

#include <vector>

#include "gnuradio-4.0/MemoryAllocators.hpp"
#include "gnuradio-4.0/Tag.hpp"

namespace {
template<typename T>
[[nodiscard]] bool is_aligned(const T* p, std::size_t a) {
    return (reinterpret_cast<std::uintptr_t>(p) % a) == 0;
}
} // namespace

const boost::ut::suite<"gr::allocator::Aligned"> _aligned = [] {
    using namespace boost::ut;

    "compile_time_traits"_test = [] {
        using A = gr::allocator::Aligned<int, 64UZ>;
        static_assert(std::allocator_traits<A>::is_always_equal::value);
        static_assert(alignof(int) <= 64UZ);
    };

    "is 64Bytes aligned"_test = [] {
        using Vec = std::vector<float, gr::allocator::Aligned<float>>;
        Vec v(128UZ);
        expect(is_aligned(v.data(), 64UZ)) << "vector data must be 64B aligned";
    };

    "page align,emt works (4096B)"_test = [] {
        using Allocator = gr::allocator::Aligned<double, 4096UZ>;
        std::vector<double, Allocator> v(1024UZ);
        expect(is_aligned(v.data(), 4096UZ)) << "vector data must be 4KB aligned";
    };

    "rebind"_test = [] {
        using Allocator       = gr::allocator::Aligned<int, 64UZ>;
        using DoubleAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<double>;
        DoubleAllocator doubleAllocator;
        double*         p = doubleAllocator.allocate(4UZ);
        expect(is_aligned(p, 64UZ));
        doubleAllocator.deallocate(p, 4UZ);
    };

    "allocate zero"_test = [] {
        using Allocator = gr::allocator::Aligned<int>;
        Allocator  a;
        int const* p = a.allocate(0UZ);
        (void)p;
    };

    "basic vector ops"_test = [] {
        using Vec = std::vector<int, gr::allocator::Aligned<int>>;
        Vec v;
        for (std::size_t i = 0UZ; i < 1000UZ; ++i) {
            v.push_back(static_cast<int>(i));
        }
        expect(eq(v.size(), 1000UZ));
        expect(eq(v.front(), 0_i));
        expect(eq(v.back(), 999_i));
        expect(is_aligned(v.data(), 64UZ));
    };
};

namespace {
struct CounterLogger {
    std::size_t alloc_count   = 0UZ;
    std::size_t dealloc_count = 0UZ;

    void operator()(const gr::allocator::detail::Event ev, [[maybe_unused]] const std::size_t count, [[maybe_unused]] const std::size_t bytes, const std::source_location, const std::string_view) {
        using enum gr::allocator::detail::Event;
        switch (ev) {
        case allocate: ++alloc_count; break;
        case deallocate: ++dealloc_count; break;
        case allocate_at_least: ++alloc_count; break;
        }
        // optional: filter or store more details
    }
};

struct CounterLoggerRef {
    using enum gr::allocator::detail::Event;

    CounterLogger* counters{};
    void           operator()(gr::allocator::detail::Event ev, std::size_t, std::size_t, std::source_location, std::string_view) const {
        switch (ev) {
        case allocate:
        case allocate_at_least: ++counters->alloc_count; break;
        case deallocate: ++counters->dealloc_count; break;
        }
    }
};
} // namespace

const boost::ut::suite<"gr::allocator::Logging"> _logging = [] {
    using namespace boost::ut;

    "default Logger wrapping Default allocator"_test = [] {
        using Alloc = gr::allocator::Logging<int>;
        std::vector<int, Alloc> v(10UZ);
        v.resize(32UZ);
        v.clear();
        v.shrink_to_fit();
        expect(is_aligned(v.data(), 64UZ));
    };

    "custom CounterLogger"_test = [] {
        using namespace boost::ut;

        using Alloc = gr::allocator::Logging<int, gr::allocator::Default<int>, CounterLogger>;
        std::vector<int, Alloc> v(0UZ, Alloc{}); // default-constructed CounterLogger inside

        v.resize(100UZ);
        v.clear();
        v.shrink_to_fit(); // not-binding behaviour mandated by the standard: https://eel.is/c%2B%2Bdraft/vector.capacity

        auto alloc_copy = v.get_allocator();
        // the exact per-call counts that get_allocator()'s copy carries after shrink_to_fit() are
        // implementation-defined and vary by stdlib (and version): libstdc++ copies the working instance,
        // libc++ may return a fresh or a working copy. only the consistency invariant is portable here;
        // the deterministic count contract is exercised by the shared-ref variant below.
        expect(alloc_copy.logger().dealloc_count <= alloc_copy.logger().alloc_count);
    };

    "custom CounterLogger w/ external ref"_test = [] {
        using namespace boost::ut;

        CounterLogger counters{};
        using Alloc = gr::allocator::Logging<int, gr::allocator::Default<int>, CounterLoggerRef>;

        { [[maybe_unused]] std::vector<int, Alloc> v(100UZ, Alloc{{}, CounterLoggerRef{&counters}}); }

        expect(counters.alloc_count >= 1UZ);
        expect(counters.dealloc_count >= 1UZ);
    };
};

struct NonTrivial {
    int               x{0};
    inline static int live{0};
    inline static int moves{0};
    inline static int copies{0};

    NonTrivial(int val = 0) noexcept : x{val} { ++live; }
    NonTrivial(NonTrivial&& o) noexcept : x{o.x} {
        ++live;
        ++moves;
        o.x = -1; // Mark as moved-from
    }
    NonTrivial(const NonTrivial& o) : x{o.x} {
        ++live;
        ++copies;
    }
    NonTrivial& operator=(NonTrivial&& o) noexcept {
        x   = o.x;
        o.x = -1;
        ++moves;
        return *this;
    }
    NonTrivial& operator=(const NonTrivial& o) {
        x = o.x;
        ++copies;
        return *this;
    }
    ~NonTrivial() { --live; }

    static void reset_counters() { live = moves = copies = 0; }
};

struct ThrowingCopy {
    inline static int copies_before_throw;
    inline static int copies_done;
    int               v{};

    ThrowingCopy(int x = 0) : v{x} {}
    // intentionally *not* noexcept so pmr::migrate uses the copy path to test exception safety for throwing copies.
    ThrowingCopy(ThrowingCopy&& o) noexcept(false) : v{o.v} {}
    ThrowingCopy(const ThrowingCopy& o) : v{o.v} {
        if (++copies_done > copies_before_throw) {
            throw std::runtime_error("Copy constructor exception");
        }
    }

    static void reset(int throw_after) {
        copies_done         = 0;
        copies_before_throw = throw_after;
    }
};

struct alignas(64) OverAligned {
    int value{42};
};

// Custom memory resource for tracking
struct CountingResource : std::pmr::memory_resource {
    std::pmr::memory_resource* upstream{std::pmr::new_delete_resource()};
    std::size_t                allocations{0};
    std::size_t                deallocations{0};
    std::size_t                bytes_allocated{0};
    std::size_t                bytes_deallocated{0};

    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        ++allocations;
        bytes_allocated += bytes;
        return upstream->allocate(bytes, alignment);
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
        ++deallocations;
        bytes_deallocated += bytes;
        upstream->deallocate(p, bytes, alignment);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }

    void reset() { allocations = deallocations = bytes_allocated = bytes_deallocated = 0; }

    bool is_balanced() const { return allocations == deallocations && bytes_allocated == bytes_deallocated; }
};

const boost::ut::suite<"gr::allocator::pmr:: ..."> _migrate = [] {
    using namespace boost::ut;
    using namespace gr::allocator;

    "migrate nullptr returns nullptr"_test = [] {
        std::pmr::memory_resource& mr1 = *std::pmr::new_delete_resource();
        std::pmr::memory_resource& mr2 = *std::pmr::new_delete_resource();

        int* p = pmr::migrate<int>(mr2, mr1, nullptr, 10);
        expect(p == nullptr);
    };

    "migrate zero count returns nullptr"_test = [] {
        std::pmr::memory_resource& mr      = *std::pmr::new_delete_resource();
        int                        data[5] = {1, 2, 3, 4, 5};

        int* p = pmr::migrate<int>(mr, mr, data, 0);
        expect(p == nullptr);
        // Original data should be untouched
        expect(eq(data[0], 1));
    };

    "migrate same resource returns original pointer"_test = [] {
        std::pmr::memory_resource& mr = *std::pmr::new_delete_resource();

        int* original = static_cast<int*>(mr.allocate(5 * sizeof(int), alignof(int)));
        std::uninitialized_fill_n(original, 5, 42);

        int* migrated = pmr::migrate<int>(mr, mr, original, 5);
        expect(migrated == original) << "Same resource should return original pointer";
        expect(eq(migrated[0], 42));

        // Clean up
        std::destroy_n(migrated, 5);
        mr.deallocate(migrated, 5 * sizeof(int), alignof(int));
    };

    "migrate trivially copyable type"_test = [] {
        CountingResource source_mr, target_mr;

        // Allocate and initialize on source
        int* source = static_cast<int*>(source_mr.allocate(5 * sizeof(int), alignof(int)));
        for (int i = 0; i < 5; ++i) {
            source[i] = i + 1;
        }

        // Migrate to target
        int* target = pmr::migrate<int>(target_mr, source_mr, source, 5);

        expect(target != nullptr);
        expect(target != source) << "Different resources should yield different pointers";

        // Verify data
        for (int i = 0; i < 5; ++i) {
            expect(eq(target[i], i + 1));
        }

        // Verify resource tracking
        expect(eq(source_mr.allocations, 1u));
        expect(eq(source_mr.deallocations, 1u));
        expect(eq(target_mr.allocations, 1u));
        expect(eq(target_mr.deallocations, 0u));

        // Clean up
        target_mr.deallocate(target, 5 * sizeof(int), alignof(int));
        expect(target_mr.is_balanced());
    };

    "migrate non-trivial type with move constructor"_test = [] {
        CountingResource source_mr, target_mr;
        NonTrivial::reset_counters();

        // Allocate and initialize on source
        NonTrivial* source = static_cast<NonTrivial*>(source_mr.allocate(3 * sizeof(NonTrivial), alignof(NonTrivial)));
        for (int i = 0; i < 3; ++i) {
            std::construct_at(source + i, i + 10);
        }
        expect(eq(NonTrivial::live, 3));
        expect(eq(NonTrivial::moves, 0));

        // Migrate to target
        NonTrivial* target = pmr::migrate<NonTrivial>(target_mr, source_mr, source, 3);

        // Should have moved, not copied (nothrow move constructor)
        expect(eq(NonTrivial::moves, 3));
        expect(eq(NonTrivial::copies, 0));
        expect(eq(NonTrivial::live, 3)) << "Same number alive after migration";

        // Verify values
        expect(eq(target[0].x, 10));
        expect(eq(target[1].x, 11));
        expect(eq(target[2].x, 12));

        // Clean up
        std::destroy_n(target, 3);
        target_mr.deallocate(target, 3 * sizeof(NonTrivial), alignof(NonTrivial));

        expect(eq(NonTrivial::live, 0));
        expect(source_mr.is_balanced());
        expect(target_mr.is_balanced());
    };

    "migrate with throwing copy constructor - exception safety"_test = [] {
        CountingResource source_mr, target_mr;

        ThrowingCopy* source = static_cast<ThrowingCopy*>(source_mr.allocate(5 * sizeof(ThrowingCopy), alignof(ThrowingCopy)));
        for (int i = 0; i < 5; ++i) {
            std::construct_at(source + i, i + 100);
        }

        ThrowingCopy::reset(2); // set to throw on 3rd copy
        expect(throws([&] {     //
            [[maybe_unused]] auto* target = pmr::migrate<ThrowingCopy>(target_mr, source_mr, source, 5UZ);
        })) << "Should throw on 3rd copy";

        expect(eq(source[0].v, 100));
        expect(eq(source[4].v, 104));

        // target resource should have cleaned up
        expect(eq(target_mr.allocations, 1u));
        expect(eq(target_mr.deallocations, 1u));
        expect(target_mr.is_balanced());

        // clean up source
        std::destroy_n(source, 5);
        source_mr.deallocate(source, 5 * sizeof(ThrowingCopy), alignof(ThrowingCopy));
        expect(source_mr.is_balanced());
    };

    "migrate over-aligned type"_test = [] {
        CountingResource source_mr, target_mr;

        OverAligned* source = static_cast<OverAligned*>(source_mr.allocate(2 * sizeof(OverAligned), alignof(OverAligned)));
        std::construct_at(source, OverAligned{100});
        std::construct_at(source + 1, OverAligned{200});

        expect(eq(reinterpret_cast<std::uintptr_t>(source) % alignof(OverAligned), 0u)) << "check alignment of source";

        OverAligned* target = pmr::migrate<OverAligned>(target_mr, source_mr, source, 2);
        expect(eq(reinterpret_cast<std::uintptr_t>(target) % alignof(OverAligned), 0u)) << "check alignment of target";
        expect(eq(target[0].value, 100));
        expect(eq(target[1].value, 200));

        // clean up
        std::destroy_n(target, 2);
        target_mr.deallocate(target, 2 * sizeof(OverAligned), alignof(OverAligned));

        expect(source_mr.is_balanced());
        expect(target_mr.is_balanced());
    };

    "migrate large array performance"_test = [] {
        CountingResource      source_mr, target_mr;
        constexpr std::size_t size = 10000;

        double* source = static_cast<double*>(source_mr.allocate(size * sizeof(double), alignof(double)));
        for (std::size_t i = 0; i < size; ++i) {
            source[i] = static_cast<double>(i) * 3.14;
        }

        // migrate (should use memcpy for trivially copyable)
        double* target = pmr::migrate<double>(target_mr, source_mr, source, size);
        expect(target != nullptr);
        expect(eq(target[0], 0.0));
        expect(eq(target[100], 100.0 * 3.14));
        expect(eq(target[size - 1], static_cast<double>(size - 1) * 3.14));

        // clean up
        target_mr.deallocate(target, size * sizeof(double), alignof(double));

        expect(source_mr.is_balanced());
        expect(target_mr.is_balanced());
    };

    "migrate between custom resources with tracking"_test = [] {
        CountingResource mr1;
        CountingResource mr2;
        CountingResource mr3;

        // create chain of migrations: mr1 -> mr2 -> mr3
        int* p1 = static_cast<int*>(mr1.allocate(4 * sizeof(int), alignof(int)));
        std::uninitialized_fill_n(p1, 4, 42);

        int* p2 = pmr::migrate<int>(mr2, mr1, p1, 4);
        expect(mr1.is_balanced());
        expect(!mr2.is_balanced()) << "mr2 has allocation but no deallocation yet";

        int* p3 = pmr::migrate<int>(mr3, mr2, p2, 4);
        expect(mr2.is_balanced());
        expect(!mr3.is_balanced());

        // final cleanup
        mr3.deallocate(p3, 4 * sizeof(int), alignof(int));

        expect(mr1.is_balanced());
        expect(mr2.is_balanced());
        expect(mr3.is_balanced());
    };
};

const boost::ut::suite<"PMR conversion helpers"> _pmrConversion = [] {
    using namespace boost::ut;
    using namespace std::string_view_literals;

    "to_pmr string round-trip"_test = [] {
        auto* mr     = std::pmr::get_default_resource();
        auto  pmrStr = gr::to_pmr("hello"sv, mr);
        expect(eq(std::string_view(pmrStr), "hello"sv));
        auto stdStr = gr::to_std(std::string_view(pmrStr));
        expect(eq(std::string_view(stdStr), "hello"sv));
    };

    "to_pmr string with custom resource"_test = [] {
        std::array<std::byte, 256>          buf{};
        std::pmr::monotonic_buffer_resource mr(buf.data(), buf.size());
        auto                                pmrStr = gr::to_pmr("custom"sv, &mr);
        expect(eq(std::string_view(pmrStr), "custom"sv));
    };

    "to_pmr/to_std vector round-trip"_test = [] {
        auto*              mr     = std::pmr::get_default_resource();
        std::vector<float> stdVec = {1.f, 2.f, 3.f};
        auto               pmrVec = gr::to_pmr(stdVec, mr);
        expect(eq(pmrVec.size(), 3UZ));
        expect(eq(pmrVec[2], 3.f));

        auto backToStd = gr::to_std(pmrVec);
        expect(eq(backToStd.size(), 3UZ));
        expect(eq(backToStd[0], 1.f));
    };

    "to_pmr vector with custom resource"_test = [] {
        std::array<std::byte, 4096>         buf{};
        std::pmr::monotonic_buffer_resource mr(buf.data(), buf.size());
        std::vector<int>                    src    = {10, 20, 30};
        auto                                pmrVec = gr::to_pmr(src, &mr);
        expect(eq(pmrVec.size(), 3UZ));
        expect(eq(pmrVec[1], 20));
    };

    "to_pmr/to_std empty containers"_test = [] {
        auto* mr = std::pmr::get_default_resource();
        expect(gr::to_pmr(std::string_view{}, mr).empty());
        expect(gr::to_std(std::string_view{}).empty());
        expect(gr::to_pmr(std::vector<int>{}, mr).empty());
        expect(gr::to_std(std::pmr::vector<int>{}).empty());
    };

    "PmrMigratable concept"_test = [] {
        expect(gr::PmrMigratable<std::pmr::string>);
        expect(gr::PmrMigratable<std::pmr::vector<float>>);
        expect(gr::PmrMigratable<std::pmr::vector<int>>);
        expect(!gr::PmrMigratable<std::string>);
        expect(!gr::PmrMigratable<std::vector<float>>);
        expect(!gr::PmrMigratable<int>);
    };

    "migrateField moves pmr::vector to new resource"_test = [] {
        std::array<std::byte, 8192>         buf{};
        std::pmr::monotonic_buffer_resource targetMr(buf.data(), buf.size());

        std::pmr::vector<float> v = {1.f, 2.f, 3.f};
        expect(v.get_allocator().resource() == std::pmr::get_default_resource());

        gr::migrateField(v, &targetMr);
        expect(v.get_allocator().resource() == &targetMr) << "resource must be rebound";
        expect(eq(v.size(), 3UZ));
        expect(eq(v[0], 1.f));
        expect(eq(v[2], 3.f));
    };

    "migrateField moves pmr::string to new resource"_test = [] {
        std::array<std::byte, 4096>         buf{};
        std::pmr::monotonic_buffer_resource targetMr(buf.data(), buf.size());

        std::pmr::string s("hello world — long enough to avoid SSO");
        gr::migrateField(s, &targetMr);
        expect(s.get_allocator().resource() == &targetMr);
        expect(eq(std::string_view(s), "hello world — long enough to avoid SSO"sv));
    };

    "set_default_resource redirects property_map allocations"_test = [] {
        std::array<std::byte, 65536>        buf{};
        std::pmr::monotonic_buffer_resource poolMr(buf.data(), buf.size());

        auto* previous = std::pmr::set_default_resource(&poolMr);

        gr::property_map map;
        map["sample_rate"]  = 48000.f;
        map["signal_name"]  = std::pmr::string("test");
        map["trigger_time"] = std::uint64_t(123456789);

        expect(map.get_allocator().resource() == &poolMr) << "property_map must use the overridden default resource";

        gr::Tag testTag{42UZ, std::move(map)};
        expect(testTag.map.get_allocator().resource() == &poolMr) << "Tag::map must use the overridden default resource";

        std::pmr::set_default_resource(previous);
    };
};

const boost::ut::suite<"gr::pmr::StaticArenaResource"> _staticArena = [] {
    using namespace boost::ut;
    using gr::pmr::OwnedStaticArenaResource;
    using gr::pmr::StaticArenaResource;

    "fresh: used=0, available=capacity"_test = [] {
        OwnedStaticArenaResource<1024UZ> arena;
        expect(eq(arena.used(), 0UZ));
        expect(eq(arena.capacity(), 1024UZ));
        expect(eq(arena.available(), 1024UZ));
    };

    "available() = capacity - used after allocations"_test = [] {
        OwnedStaticArenaResource<1024UZ> arena;
        std::ignore = arena.allocate(100UZ, 1UZ);
        expect(eq(arena.available(), arena.capacity() - arena.used()));
        std::ignore = arena.allocate(64UZ, 8UZ);
        expect(eq(arena.available(), arena.capacity() - arena.used()));
    };

    "allocate advances _used by bytes + alignment padding"_test = [] {
        OwnedStaticArenaResource<1024UZ, 16UZ> arena;
        void*                                  p1 = arena.allocate(13UZ, 1UZ);
        expect(p1 != nullptr);
        expect(eq(arena.used(), 13UZ));
        void* p2 = arena.allocate(8UZ, 8UZ);
        expect(p2 != nullptr);
        expect(eq(arena.used() % 8UZ, 0UZ)) << "second alloc starts 8-byte aligned";
        expect(arena.used() >= 13UZ + 8UZ);
    };

    "zero-byte allocate does not consume padding"_test = [] {
        OwnedStaticArenaResource<1024UZ, 1UZ> arena;
        const std::size_t                     before = arena.used();
        void*                                 p1     = arena.allocate(0UZ, 64UZ);
        void*                                 p2     = arena.allocate(0UZ, 64UZ);
        expect(p1 != nullptr) << "zero-byte alloc must return a usable address";
        expect(p2 != nullptr);
        expect(eq(arena.used(), before)) << "zero-byte alloc must not consume arena bytes";
    };

    "returned pointer respects requested alignment"_test = [] {
        OwnedStaticArenaResource<4096UZ, 64UZ> arena;
        for (std::size_t align : {1UZ, 8UZ, 16UZ, 32UZ, 64UZ}) {
            void* p = arena.allocate(7UZ, align);
            expect(std::bit_cast<std::uintptr_t>(p) % align == 0UZ) << std::format("align={}", align);
        }
    };

    "deallocate is no-op (monotonic)"_test = [] {
        OwnedStaticArenaResource<1024UZ> arena;
        void*                            p   = arena.allocate(32UZ, 8UZ);
        const std::size_t                pre = arena.used();
        arena.deallocate(p, 32UZ, 8UZ);
        expect(eq(arena.used(), pre)) << "monotonic resource does not reclaim";
    };

    "reset() zeros _used"_test = [] {
        OwnedStaticArenaResource<1024UZ> arena;
        std::ignore = arena.allocate(256UZ, 8UZ);
        expect(gt(arena.used(), 0UZ));
        arena.reset();
        expect(eq(arena.used(), 0UZ));
        expect(eq(arena.available(), 1024UZ));
    };

    "exhaustion panics with gr::exception carrying 'exhausted'; arena state unchanged"_test = [] {
        OwnedStaticArenaResource<128UZ> arena;
        const std::size_t               before = arena.used();
        try {
            std::ignore = arena.allocate(512UZ, 1UZ);
            expect(false) << "allocate should have panicked";
        } catch (const gr::exception& e) {
            expect(std::string_view{e.message}.contains("exhausted"));
        }
        expect(eq(arena.used(), before)) << "panic must not mutate arena state";
    };

    "do_is_equal: same instance true, distinct instances false"_test = [] {
        OwnedStaticArenaResource<256UZ> a;
        OwnedStaticArenaResource<256UZ> b;
        expect(a.is_equal(a));
        expect(!a.is_equal(b));
    };

    "interop: std::pmr::vector<int> with reserve pre-allocates once"_test = [] {
        OwnedStaticArenaResource<2048UZ> arena;
        std::pmr::vector<int>            v{&arena};
        v.reserve(64UZ);
        const std::size_t afterReserve = arena.used();
        expect(eq(afterReserve, 64UZ * sizeof(int))) << "reserve(64) allocates exactly one int[64]";
        for (int i = 0; i < 64; ++i) {
            v.push_back(i);
        }
        expect(eq(v.size(), 64UZ));
        expect(eq(arena.used(), afterReserve)) << "no reallocation during push_back inside reserved capacity";
        expect(eq(v[63UZ], 63));
    };

    "composition with CountingResource: arena.used() and counter.liveBytes agree on alloc"_test = [] {
        OwnedStaticArenaResource<4096UZ>     arena;
        gr::allocator::pmr::CountingResource counter;
        counter.upstream = &arena;
        void* p          = counter.allocate(128UZ, 8UZ);
        expect(p != nullptr);
        expect(eq(counter.liveBytes, 128UZ));
        expect(arena.used() >= 128UZ);
        counter.deallocate(p, 128UZ, 8UZ); // counter decrements liveBytes; arena.used() unchanged (monotonic)
        expect(eq(counter.liveBytes, 0UZ));
        expect(arena.used() >= 128UZ) << "arena is monotonic — bytes stay accounted as in-use";
    };

    "external storage: span over caller-owned buffer"_test = [] {
        alignas(16UZ) std::byte        buf[512UZ]{};
        StaticArenaResource            arena{std::span<std::byte>{buf}};
        std::pmr::vector<std::int32_t> v{&arena};
        v.reserve(16UZ);
        for (std::int32_t i = 0; i < 16; ++i) {
            v.push_back(i);
        }
        expect(eq(v.size(), 16UZ));
        expect(gt(arena.used(), 0UZ));
        expect(le(arena.used(), 512UZ));
    };

    "over-alignment (alignment > kAlignment): padded transparently"_test = [] {
        OwnedStaticArenaResource<4096UZ, 8UZ> arena;
        void*                                 p = arena.allocate(16UZ, 64UZ);
        expect(std::bit_cast<std::uintptr_t>(p) % 64UZ == 0UZ);
    };

    "over-alignment that overflows: panics"_test = [] {
        // 64-byte-aligned raw[], span starts at +1 so base is provably never 64-aligned;
        // 64-byte alignment then forces ≥ 63 padding + 16 bytes > 32-byte capacity → panic.
        alignas(64UZ) std::byte raw[80UZ]{};
        StaticArenaResource     arena{std::span<std::byte>{raw + 1UZ, 32UZ}};
        try {
            std::ignore = arena.allocate(16UZ, 64UZ);
            expect(false) << "expected exhaustion panic";
        } catch (const gr::exception& e) {
            expect(std::string_view{e.message}.contains("exhausted"));
        }
    };

    "external storage: allocate past span end panics"_test = [] {
        std::byte           buf[16UZ]{};
        StaticArenaResource arena{std::span<std::byte>{buf}};
        try {
            std::ignore = arena.allocate(32UZ, 1UZ); // request > span capacity
            expect(false) << "expected exhaustion panic";
        } catch (const gr::exception& e) {
            expect(std::string_view{e.message}.contains("exhausted"));
        }
        expect(eq(arena.used(), 0UZ));
    };
};

int main() { /* not needed for UT */ }
