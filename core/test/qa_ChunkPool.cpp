#include <boost/ut.hpp>

#include <gnuradio-4.0/ChunkPool.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

using namespace boost::ut;

namespace {
[[nodiscard]] bool isAligned(std::span<std::byte> chunk, std::size_t alignment) noexcept {
    return (reinterpret_cast<std::uintptr_t>(chunk.data()) % alignment) == 0UZ; // NOSONAR — alignment probe
}
} // namespace

const boost::ut::suite<"ChunkPool"> _chunkPool = [] {
    "acquire yields an aligned chunk of chunkBytes"_test = [] {
        gr::allocator::pmr::CountingResource upstream;
        gr::ChunkPool                        pool(&upstream, 256UZ, 64UZ);

        std::span<std::byte> c = pool.acquire();
        expect(eq(c.size(), 256UZ));
        expect(isAligned(c, 64UZ));
        expect(eq(pool.residentChunks(), 1UZ));
        expect(eq(upstream.allocCount, 1UZ));
        pool.release(c);
    };

    "release then acquire reuses the same chunk without growing upstream"_test = [] {
        gr::allocator::pmr::CountingResource upstream;
        gr::ChunkPool                        pool(&upstream, 256UZ);

        std::span<std::byte> a = pool.acquire();
        pool.release(a);
        std::span<std::byte> b = pool.acquire();

        expect(eq(b.data(), a.data())); // LIFO intrusive free-list hands the same chunk back
        expect(eq(pool.residentChunks(), 1UZ));
        expect(eq(upstream.allocCount, 1UZ)); // no second upstream allocation
        pool.release(b);
    };

    "grow-then-settle: N acquires grow resident to N, releases fill the free-list"_test = [] {
        gr::allocator::pmr::CountingResource upstream;
        gr::ChunkPool                        pool(&upstream, 256UZ);

        std::array<std::span<std::byte>, 5> chunks{};
        for (auto& c : chunks) {
            c = pool.acquire();
        }
        expect(eq(pool.residentChunks(), 5UZ));
        expect(eq(pool.freeChunks(), 0UZ));
        expect(eq(upstream.allocCount, 5UZ));

        for (auto& c : chunks) {
            pool.release(c);
        }
        expect(eq(pool.freeChunks(), 5UZ));
        expect(eq(pool.residentChunks(), 5UZ)); // still resident — held on the free-list
        expect(eq(upstream.deallocCount, 0UZ)); // not yet returned upstream
    };

    "reclaimToUpstream returns idle chunks; keepResident is the floor"_test = [] {
        gr::allocator::pmr::CountingResource upstream;
        gr::ChunkPool                        pool(&upstream, 256UZ);

        std::array<std::span<std::byte>, 6> chunks{};
        for (auto& c : chunks) {
            c = pool.acquire();
        }
        for (auto& c : chunks) {
            pool.release(c);
        }

        pool.reclaimToUpstream(2UZ); // keep 2 resident, return the rest
        expect(eq(pool.freeChunks(), 2UZ));
        expect(eq(pool.residentChunks(), 2UZ));
        expect(eq(upstream.deallocCount, 4UZ));

        pool.reclaimToUpstream(0UZ); // return everything → RAM available elsewhere
        expect(eq(pool.freeChunks(), 0UZ));
        expect(eq(pool.residentChunks(), 0UZ));
        expect(eq(upstream.deallocCount, 6UZ));
        expect(eq(upstream.liveBytes, 0UZ));
    };

    "maxChunks cap backpressures with an empty span (no exception)"_test = [] {
        gr::allocator::pmr::CountingResource upstream;
        gr::ChunkPool                        pool(&upstream, 256UZ, 64UZ, 2UZ); // hard cap = 2

        std::span<std::byte> a = pool.acquire();
        std::span<std::byte> b = pool.acquire();
        std::span<std::byte> c = pool.acquire(); // over cap

        expect(!a.empty());
        expect(!b.empty());
        expect(c.empty()); // exception-free backpressure
        expect(eq(pool.residentChunks(), 2UZ));

        pool.release(b);
        std::span<std::byte> d = pool.acquire(); // a freed slot is available again
        expect(!d.empty());
        pool.release(a);
        pool.release(d);
    };

    "destructor returns the free-list to upstream (no leak)"_test = [] {
        gr::allocator::pmr::CountingResource upstream;
        {
            gr::ChunkPool                       pool(&upstream, 256UZ);
            std::array<std::span<std::byte>, 3> chunks{};
            for (auto& c : chunks) {
                c = pool.acquire();
            }
            for (auto& c : chunks) {
                pool.release(c);
            }
        } // pool destroyed with 3 chunks on the free-list
        expect(eq(upstream.allocCount, 3UZ));
        expect(eq(upstream.deallocCount, 3UZ));
        expect(eq(upstream.liveBytes, 0UZ));
    };
};

int main() { return 0; }
