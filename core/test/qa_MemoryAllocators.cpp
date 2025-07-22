#include <boost/ut.hpp>

#include <vector>

#include "gnuradio-4.0/MemoryAllocators.hpp"

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
        expect(alloc_copy.logger().alloc_count >= 1UZ);
#if defined(_LIBCPP_VERSION) // libc++ -- specific behaviour
        expect(alloc_copy.logger().dealloc_count == 1UZ);
#else // libstdc++ -- specific behaviour
        expect(alloc_copy.logger().dealloc_count == 0UZ); // nothing deallocated here - vector and allocator still lives
#endif
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

int main() { /* not needed for UT */ }
