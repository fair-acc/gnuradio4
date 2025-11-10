#include <boost/ut.hpp>

#include <gnuradio-4.0/PmtCollections.hpp>

struct NonTrivial {
    int               x{0};
    inline static int live{0};
    NonTrivial(int x_ = 0) noexcept : x(x_) { ++live; }
    NonTrivial(NonTrivial&& o) noexcept : x{o.x} { ++live; }
    NonTrivial& operator=(NonTrivial&& o) noexcept {
        x = o.x;
        return *this;
    }
    ~NonTrivial() { --live; }
};

struct CopyableNT {
    int               x{0};
    inline static int live{0};
    CopyableNT() noexcept { ++live; }
    CopyableNT(const CopyableNT& o) : x{o.x} { ++live; } // copyable
    CopyableNT(CopyableNT&& o) noexcept : x{o.x} { ++live; }
    CopyableNT& operator=(const CopyableNT& o) {
        x = o.x;
        return *this;
    }
    CopyableNT& operator=(CopyableNT&& o) noexcept {
        x = o.x;
        return *this;
    }
    ~CopyableNT() { --live; }
};

struct ThrowCopy {
    static inline std::size_t copies_before_throw = ~0UZ;
    static inline std::size_t copies_done         = 0UZ;
    int                       v{};
    ThrowCopy() = default;
    ThrowCopy(int x) : v{x} {}
    ThrowCopy(const ThrowCopy& o) : v{o.v} {
        if (++copies_done > copies_before_throw) {
            throw std::runtime_error{"boom"};
        }
    }
    ThrowCopy& operator=(const ThrowCopy&) = delete;
    ThrowCopy(ThrowCopy&&) noexcept        = default;
    ThrowCopy& operator=(ThrowCopy&&)      = delete;
};

struct MoveOnly {
    MoveOnly()                               = default;
    MoveOnly(MoveOnly&&) noexcept            = default;
    MoveOnly& operator=(MoveOnly&&) noexcept = default;
    MoveOnly(const MoveOnly&)                = delete;
    MoveOnly& operator=(const MoveOnly&)     = delete;
    int       x{};
};

const boost::ut::suite<"gr::pmr::vector"> _pmr = [] {
    using namespace boost::ut;
    static_assert(!std::is_trivially_copyable_v<gr::pmr::vector<double, true>>);
    static_assert(std::is_trivially_copyable_v<gr::pmr::vector<double, false>>);

    "pmr::vector<T> - unmanaged is_trivially_copyable"_test = [] {
        using U = gr::pmr::vector<int, false>;
        expect(std::is_trivially_copyable_v<U>) << "must be trivially copyable";
        U v;
        expect(eq(v.size(), 0u));
        int buf[4]{1, 2, 3, 4};
        v.attach(buf, 4);
        expect(eq(v.front(), 1) && eq(v.back(), 4));
        auto* p = v.release();
        expect(p == buf) << "release should not free";
        expect(eq(v.size(), 0u));
    };

    "pmr::vector<T> - managed resize->preserves->prefix"_test = [] {
        gr::pmr::vector<int, true> v(3);
        for (std::size_t i = 0; i < 3; ++i) {
            v[i] = static_cast<int>(i) + 1;
        }
        v.resize(6UZ);
        expect(eq(v.size(), 6u));
        expect(eq(v[0], 1) && eq(v[2], 3));
        for (std::size_t i = 3; i < 6; ++i) {
            expect(eq(v[i], 0));
        }
        v.resize(2UZ);
        expect(eq(v.size(), 2u));
        expect(eq(v[0], 1) && eq(v[1], 2));
        v.clear();
        expect(eq(v.size(), 0u));
    };

    "pmr::vector<T> - managed nontrivial lifetimes"_test = [] {
        expect(eq(NonTrivial::live, 0));
        gr::pmr::vector<NonTrivial, true> v(5);
        expect(eq(NonTrivial::live, 5));
        v[0].x = 42;
        v.resize(2UZ);
        expect(eq(NonTrivial::live, 2));
        expect(eq(v[0].x, 42));
        v.clear();
        expect(eq(NonTrivial::live, 0));
    };

    "pmr::vector<T> - comparisons lexicographical"_test = [] {
        gr::pmr::vector<int, true> a(3);
        gr::pmr::vector<int, true> b(3);
        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        b[0] = 1;
        b[1] = 2;
        b[2] = 3;
        expect(a == b);
        b.resize(4);
        b[3] = 0;
        expect((a <=> b) == std::strong_ordering::less);
    };

    struct counting_resource : std::pmr::memory_resource {
        std::pmr::memory_resource* upstream{std::pmr::new_delete_resource()};
        std::size_t                a{0}, d{0}, bytes{0};
        void*                      do_allocate(std::size_t n, std::size_t align) override {
            ++a;
            bytes += n;
            return upstream->allocate(n, align);
        }
        void do_deallocate(void* p, std::size_t n, std::size_t align) override {
            ++d;
            bytes -= n;
            upstream->deallocate(p, n, align);
        }
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
    };

    "pmr::vector<T,true> iterator-ctor from pmr::vector iterators"_test = [] {
        counting_resource     mr;
        std::pmr::vector<int> pv{&mr};
        pv.emplace_back(1);
        pv.emplace_back(2);
        pv.emplace_back(3);
        gr::pmr::vector<int, true> v1(pv.begin(), pv.end(), &mr);
        expect(eq(v1.size(), 3u));
        expect(eq(v1[0], 1) and eq(v1[1], 2) and eq(v1[2], 3));
    };

    "pmr::vector<T,true> span-ctor dispatches to iterator-ctor"_test = [] {
        counting_resource          mr;
        int                        arr[]{4, 5, 6};
        gr::pmr::vector<int, true> v(std::span<const int>(arr), &mr);
        expect(eq(v.size(), 3u));
        expect(eq(v[0], 4) and eq(v[1], 5) and eq(v[2], 6));
    };

    "pmr::vector<T,true> generic-range-ctor does not hijack copy-ctor"_test = [] {
        counting_resource          mr;
        gr::pmr::vector<int, true> v1({7, 8, 9}, &mr);
        gr::pmr::vector<int, true> v2(v1); // should bind to copy-ctor
        expect(eq(v2.size(), 3u));
        expect(eq(v2[0], 7) and eq(v2[1], 8) and eq(v2[2], 9));
    };

    "pmr::vector<ThrowCopy,true> iterator-ctor throws & frees on partial copy"_test = [] {
        counting_resource mr;

        std::vector<ThrowCopy> src;
        src.emplace_back(1);
        src.emplace_back(2);
        src.emplace_back(3);
        ThrowCopy::copies_done         = 0;
        ThrowCopy::copies_before_throw = 2; // throw on 3rd copy

        gr::pmr::vector<ThrowCopy, true> guard(1, &mr);
        guard[0].v    = 42;
        const auto a0 = mr.a, d0 = mr.d, b0 = mr.bytes;

        expect(throws([&] { [[maybe_unused]] gr::pmr::vector<ThrowCopy, true> will_throw(std::cbegin(src), std::cend(src), &mr); }));

        // one alloc+free for the failed construction; existing object intact
        expect(eq(mr.a, a0 + 1));
        expect(eq(mr.d, d0 + 1));
        expect(eq(mr.bytes, b0));
        expect(eq(guard.size(), 1u));
        expect(eq(guard[0].v, 42));
    };

    "pmr::vector<T> - cross-resource copy"_test = [] {
        counting_resource          mr1, mr2;
        gr::pmr::vector<int, true> v1({1, 2, 3}, &mr1);
        gr::pmr::vector<int, true> v2(v1);
        gr::pmr::vector<int, true> v3(&mr2);
        v3 = v1;
        expect(v3.resource() == &mr2) << "retains its resource";
    };

    "pmr::vector<T> ctor default & sized"_test = [] {
        counting_resource          mr;
        gr::pmr::vector<int, true> a; // default
        expect(eq(a.size(), 0u));

        gr::pmr::vector<int, true> b(5UZ, &mr);
        expect(eq(b.size(), 5UZ)) << "allocates 5*sizeof(int)";
        for (auto& x : b) {
            expect(eq(x, 0)); // value-initialised tail
        }

        gr::pmr::vector<int, true> c(4, 7, &mr); // fill
        expect(eq(c.size(), 4u));
        for (auto& x : c) {
            expect(eq(x, 7));
        }
        c.clear();
        expect(eq(mr.bytes, 5u * sizeof(int))); // b still alive
    };

    "pmr::vector<T> ctor from iter span ilist range"_test = [] {
        counting_resource mr;

        std::pmr::vector<int> pv{&mr};
        pv.assign({1, 2, 3, 4});
        gr::pmr::vector<int, true> v1(pv.begin(), pv.end(), &mr);
        expect(eq(v1.size(), 4u) && eq(v1[0], 1) && eq(v1[3], 4)) << "iterator pair";

        std::array<int, 3>         arr{9, 8, 7};
        gr::pmr::vector<int, true> v2(std::span<const int>(arr), &mr);
        expect(eq(v2.size(), 3u) && eq(v2[0], 9) && eq(v2[2], 7)) << "span";

        gr::pmr::vector<int, true> v3({5, 6}, &mr);
        expect(eq(v3.size(), 2u) && eq(v3[1], 6)) << "initializer_list";

        gr::pmr::vector<int, true> v4(std::views::iota(0, 3), &mr);
        expect(eq(v4.size(), 3u) && eq(v4[2], 2)) << "range (views)";

        v4.assign(std::span<const int>(arr));
        expect(eq(v4.size(), 3u) && eq(v4[1], 8)) << "assign variant #1";

        v4.assign({11, 12, 13, 14});
        expect(eq(v4.size(), 4u) && eq(v4[0], 11) && eq(v4[3], 14)) << "assign variant #2";

        v4.assign(pv.begin(), pv.end());
        expect(eq(v4.size(), 4u) && eq(v4[2], 3)) << "assign variant #3";
    };

    "pmr::vector<T> copy-ctor copy->assign->move-ctor move-assign self-assign"_test = [] {
        counting_resource mr;

        gr::pmr::vector<int, true> a({1, 2, 3, 4}, &mr);
        gr::pmr::vector<int, true> b(a); // copy ctor
        expect(a == b);

        gr::pmr::vector<int, true> c(1, &mr);
        c = b; // copy assign
        expect(c == b);

        gr::pmr::vector<int, true> d(std::move(b)); // move ctor
        expect(d == a);
        expect(eq(b.size(), 0u)); // moved-from

        gr::pmr::vector<int, true> e(2, 9, &mr);
        e = std::move(d); // move assign
        expect(e == a);
        expect(eq(d.size(), 0u));

        // test self-assignment
#if defined(__clang__) or defined(__EMSCRIPTEN__) or defined(__GNUC__)
#pragma GCC diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#pragma GCC diagnostic ignored "-Wself-move"
#endif
        e = e;
        expect(e == a) << "self-assign (copy)";
        e = std::move(e);
        expect(e == a) << "self-assign (move)";
#if defined(__clang__) or defined(__EMSCRIPTEN__) or defined(__GNUC__)
#pragma GCC diagnostic pop
#pragma clang diagnostic pop
#endif
    };

    "pmr::vector<T> resize preserves prefix & clear deallocates"_test = [] {
        counting_resource          mr;
        gr::pmr::vector<int, true> v(3, &mr);
        v[0] = 1;
        v[1] = 2;
        v[2] = 3;

        v.resize(6);
        expect(eq(v.size(), 6u));
        expect(eq(v[0], 1) && eq(v[2], 3));
        for (std::size_t i = 3; i < 6; ++i) {
            expect(eq(v[i], 0));
        }

        v.resize(2);
        expect(eq(v.size(), 2u) && eq(v[0], 1) && eq(v[1], 2));

        const auto a_before = mr.a;
        const auto d_before = mr.d;
        v.clear();
        expect(eq(v.size(), 0u));
        expect(eq(mr.bytes, 0u));
        expect(mr.a >= a_before && mr.d == d_before + 1);
    };

    "pmr::vector<T> nontrivial lifetimes in ctors & assign"_test = [] {
        counting_resource mr;
        expect(eq(CopyableNT::live, 0));

        gr::pmr::vector<CopyableNT, true> v1(5, &mr);
        expect(eq(CopyableNT::live, 5)) << "default-constructed";

        gr::pmr::vector<CopyableNT, true> v2(v1);
        expect(eq(CopyableNT::live, 10)) << "copy ctor";

        gr::pmr::vector<CopyableNT, true> v3(1, &mr);
        v3 = v1;
        // rough sanity: v1(5) + v2(5) + v3(5) - old v3(1) = 14
        expect(eq(CopyableNT::live, 15)) << "copy assign";

        v1.clear();
        v2.clear();
        v3.clear();
        expect(eq(CopyableNT::live, 0));
        expect(eq(mr.bytes, 0u));
    };

    "pmr::vector<T> strong guarantee ctor from iter throws"_test = [] {
        counting_resource mr;

        std::vector<ThrowCopy> src;
        src.emplace_back(1);
        src.emplace_back(2);
        src.emplace_back(3);

        ThrowCopy::copies_done         = 0UZ;
        ThrowCopy::copies_before_throw = 2UZ; // throw on 3rd copy

        // build a 'target' first so we can check it doesn't leak/change on failed construction
        gr::pmr::vector<ThrowCopy, true> target(1, &mr);
        target[0].v   = 99;
        const auto a0 = mr.a, d0 = mr.d, bytes0 = mr.bytes;

        expect(throws([&] { [[maybe_unused]] gr::pmr::vector<ThrowCopy, true> will_throw(std::cbegin(src), std::cend(src), &mr); })) << "should throw on 3rd copy";

        // one alloc + one free during the failed construction; no outstanding bytes
        expect(eq(mr.a, a0 + 1));
        expect(eq(mr.d, d0 + 1));
        expect(eq(mr.bytes, bytes0));

        // existing object intact
        expect(eq(target.size(), 1u));
        expect(eq(target[0].v, 99));
        target.clear();
    };

    "pmr::vector<T,true> push/emplace/pop & capacity"_test = [] {
        counting_resource          mr;
        gr::pmr::vector<int, true> v(&mr);

        expect(eq(v.size(), 0u));
        expect(eq(v.capacity(), 0u));

        v.reserve(4);
        expect(eq(v.capacity(), 4u));

        // No extra allocs within capacity
        const auto a0 = mr.a, d0 = mr.d, b0 = mr.bytes;

        v.push_back(1);
        v.push_back(2);
        auto& r = v.emplace_back(3);
        expect(eq(r, 3));
        expect(eq(v.size(), 3u));
        expect(eq(v.capacity(), 4u));
        expect(eq(mr.a, a0));
        expect(eq(mr.d, d0));
        expect(eq(mr.bytes, b0)); // still holding the 4*int buffer

        v.pop_back();
        expect(eq(v.size(), 2u));
        expect(eq(v[0], 1));
        expect(eq(v[1], 2));

        // shrink_to_fit reallocates down to size
        v.shrink_to_fit();
        expect(eq(v.capacity(), v.size()));
    };

    "pmr::vector<NonTrivial,true> emplace/move & lifetimes"_test = [] {
        counting_resource mr;
        NonTrivial::live = 0;

        gr::pmr::vector<NonTrivial, true> v(&mr);
        v.reserve(1);
        expect(eq(NonTrivial::live, 0));

        auto& a = v.emplace_back(7);
        expect(eq(a.x, 7));
        expect(eq(NonTrivial::live, 1));

        // move-into push_back
        v.push_back(NonTrivial{8});
        expect(eq(NonTrivial::live, 2));
        expect(eq(v.size(), 2u));

        v.pop_back(); // destroys one
        expect(eq(NonTrivial::live, 1));
        v.clear();
        expect(eq(NonTrivial::live, 0));
        expect(eq(mr.bytes, 0u));
    };

    // Non-managed: attach external storage and use in-place emplace/push/pop
    "pmr::vector<T,false> attach + in-place growth (no allocations)"_test = [] {
        counting_resource                    mr;
        std::pmr::polymorphic_allocator<int> pa{&mr};

        constexpr std::size_t CAP = 4;
        int*                  buf = pa.allocate(CAP); // raw uninitialised storage for ints

        gr::pmr::vector<int, false> u;
        u.attach(buf, /*size*/ 0, /*cap*/ CAP, &mr);

        expect(eq(u.size(), 0u));
        expect(eq(u.capacity(), CAP));

        u.emplace_back(11);
        u.push_back(22);
        expect(eq(u.size(), 2u));
        expect(eq(u[0], 11));
        expect(eq(u[1], 22));

        u.pop_back();
        expect(eq(u.size(), 1u));
        expect(eq(u[0], 11));

        // cleanup external storage (destroy remaining constructed elements)
        auto* raw = u.release();
        std::destroy_n(raw, 1);
        pa.deallocate(raw, CAP);

        // Non-managed must not have touched mr counters (only our manual allocs)
        // Nothing to assert strictly here except the test didn't throw.
    };

    "pmr::vector<T> - cross-resource copy keeps dest resource & frees old"_test = [] {
        counting_resource mrA, mrB;

        gr::pmr::vector<int, true> src({1, 2, 3, 4}, &mrA); // size=4, cap=4
        gr::pmr::vector<int, true> dst(1, &mrB);            // size=1, cap=1
        expect(eq(dst.resource(), &mrB));

        const std::size_t aA0 = mrA.a;
        const std::size_t dA0 = mrA.d;
        const std::size_t bA0 = mrA.bytes;

        dst = src; // cross-resource path

        // dst keeps mrB
        expect(eq(dst.resource(), &mrB));
        expect(eq(dst.size(), 4u));
        expect(eq(dst[0], 1) and eq(dst[1], 2) and eq(dst[2], 3) and eq(dst[3], 4));

        // mrA unchanged by dst assignment (src unchanged)
        expect(eq(mrA.a, aA0));
        expect(eq(mrA.d, dA0));
        expect(eq(mrA.bytes, bA0));

        // mrB now holds 4 ints; old 1-int block should be freed already
        expect(mrB.bytes >= 4u); // exact numbers depend on your logger/resource

        // destructor must free using capacity, not size
        dst.clear();
        src.clear();
        expect(eq(mrA.bytes, 0u));
        expect(eq(mrB.bytes, 0u));
    };

    "pmr::vector<T> - capacity tracked deallocation"_test = [] {
        counting_resource          mr;
        gr::pmr::vector<int, true> v(&mr);
        v.reserve(8); // capacity=8, size=0 (your impl should set _capacity=8)
        v.emplace_back(7);
        v.emplace_back(9);
        expect(eq(v.size(), 2u));

        // Ensure destructor/clear uses capacity (8*4), not size (2*4)
        v.clear();
        expect(eq(mr.bytes, 0u));
    };
};

int main() { /* not needed for UT */ }
