#include <boost/ut.hpp>
#include <chrono>
#include <numeric>
#include <thread>
#include <vector>

#include <gnuradio-4.0/AtomicRef.hpp>

using namespace boost::ut;
using gr::atomic_ref;
using gr::AtomicRef;

namespace {

template<typename T>
struct Aligned {
    alignas(alignof(T)) T v{};
};

} // namespace

const suite<"gr::atomic_ref"> _0 = [] {
    "basic load/store"_test = [] {
        Aligned<std::size_t> x{.v = 0UZ};
        auto                 a = atomic_ref(x.v);

        expect(eq(a.load_acquire(), 0UZ));
        a.store_release(42UZ);
        expect(eq(a.load_acquire(), 42UZ));
    };

    "compare_exchange semantics"_test = [] {
        Aligned<std::size_t> x{.v = 5};
        auto                 a = atomic_ref(x.v);

        std::size_t exp = 5UZ;
        expect(a.compare_exchange(exp, 7)); // success, x:=7
        expect(eq(x.v, 7UZ));

        exp = 5UZ;
        expect(!a.compare_exchange(exp, 9)); // fail; exp updated to current
        expect(eq(exp, 7UZ));
        expect(eq(x.v, 7UZ));
    };

    "fetch_add/sub"_test = [] {
        Aligned<std::size_t> x{.v = 10UZ};
        auto                 a = atomic_ref(x.v);

        expect(eq(a.fetch_add(3UZ), 10UZ)); // returns old
        expect(eq(a.load_acquire(), 13UZ));
        expect(eq(a.fetch_sub(2UZ), 13UZ));
        expect(eq(a.load_acquire(), 11UZ));
    };

    "wait/notify (or SYCL busy-loop) wakes"_test = [] {
        using namespace std::chrono_literals;

        Aligned<std::size_t> flag{.v = 0};
        auto                 aflag = atomic_ref(flag.v);

        std::thread t([&] {
            // wait returns once flag != 0
            aflag.wait(0UZ);
            expect(eq(aflag.load_acquire(), 1UZ));
        });

        std::this_thread::sleep_for(5ms);
        aflag.store_release(1UZ);
#if !defined(GR_HAS_SYCL)
        aflag.notify_all(); // no-op under SYCL path
#endif
        t.join();
    };

    "acquire-release ordering"_test = [] {
        // Producer writes payload then sets flag (release).
        // Consumer spins on flag (acquire) then reads payload.
        struct Shared {
            int         payload;
            std::size_t flag;
        } s{.payload = -1, .flag = 0};
        auto aflag = atomic_ref(s.flag);

        std::thread prod([&] {
            s.payload = 1234;       // plain store
            aflag.store_release(1); // publish
#if !defined(GR_HAS_SYCL)
            aflag.notify_one();
#endif
        });

        std::thread cons([&] {
            // Wait for publication
            while (aflag.load_acquire() == 0) {
#if defined(GR_HAS_SYCL)
                std::this_thread::yield(); // polite busy wait on SYCL path
#else
                aflag.wait(0);
#endif
            }
            // Must see the published payload due to acquire/release
            expect(eq(s.payload, 1234));
        });

        prod.join();
        cons.join();
    };

    "multi-thread fetch_add totals"_test = [] {
        constexpr std::size_t kThreads = 4UZ;
        constexpr std::size_t kIters   = 10'000UZ;

        Aligned<std::size_t> x{.v = 0UZ};
        auto                 a = atomic_ref(x.v);

        std::vector<std::thread> th;
        th.reserve(kThreads);
        for (std::size_t t = 0; t < kThreads; ++t) {
            th.emplace_back([&] {
                for (std::size_t i = 0; i < kIters; ++i)
                    (void)a.fetch_add(1);
            });
        }
        for (auto& t : th)
            t.join();

        expect(eq(a.load_acquire(), kThreads * kIters));
    };

    "aliasing across multiple AtomicRef views"_test = [] {
        Aligned<std::size_t>   x{.v = 0UZ};
        AtomicRef<std::size_t> a1{x.v}, a2{x.v};
        a1.store_release(7UZ);
        expect(eq(a2.load_acquire(), 7UZ));
        (void)a2.fetch_add(1UZ);
        expect(eq(a1.load_acquire(), 8UZ));
    };
};

int main() { /* not needed for UT */ }
