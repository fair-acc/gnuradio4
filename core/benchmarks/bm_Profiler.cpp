#include <benchmark.hpp>

#include <gnuradio-4.0/Profiler.hpp>

using namespace gr::profiling;

inline constexpr std::size_t N_ITER    = 7;
inline constexpr std::size_t N_SAMPLES = 1;

inline void run_without_profiler() {
    const auto start = detail::clock::now();

    long long r = 0;
    for (std::size_t i = 0; i < 1000; ++i) {
        for (std::size_t j = 0; j < 1000; ++j) {
            std::vector<int> v(10000);
            std::iota(v.begin(), v.end(), 1);
            r += std::accumulate(v.begin(), v.end(), 0);
        }
    }

    const auto elapsed = detail::clock::now() - start;
    std::print("The sum of sums is {} and it took {}ms\n", r, std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
}

template<ProfilerLike TProfiler>
inline void run_with_profiler(TProfiler& p) {
    const auto start   = detail::clock::now();
    auto&      handler = p.forThisThread();

    [[maybe_unused]] auto whole_calculation_event = handler.startCompleteEvent("whole_calculation");
    long long             r                       = 0;
    for (std::size_t i = 0; i < 1000; ++i) {
        auto async_event = handler.startAsyncEvent("iteration", {}, {{"arg1", 2}, {"arg2", "hello"}});
        for (std::size_t j = 0; j < 1000; ++j) {
            std::vector<int> v(10000);
            std::iota(v.begin(), v.end(), 1);
            r += std::accumulate(v.begin(), v.end(), 0);
            async_event.step();
        }
    }

    const auto elapsed = detail::clock::now() - start;
    std::print("The sum of sums is {} and it took {}ms\n", r, std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
}

[[maybe_unused]] inline const boost::ut::suite profiler_tests = [] {
    using namespace boost::ut;
    using namespace benchmark;

    Profiler prof;
    "default profiler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&p = prof] { run_with_profiler(p); };

    null::Profiler null_prof;
    "null profiler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&p = null_prof] { run_with_profiler(p); };

    "no profiler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [] { run_without_profiler(); };
};

int main() { /* not needed by the UT framework */ }
