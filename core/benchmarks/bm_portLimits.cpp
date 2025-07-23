#include <benchmark.hpp>
#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Profiler.hpp>

#include <algorithm>
#include <ranges>
#include <vector>

using namespace boost::ut;
using namespace benchmark;
using namespace gr;

struct TestLimits {
    std::size_t minSync;
    std::size_t maxSync;
    std::size_t maxAvail;
    bool        hasAsync;
};

struct BenchBlock : Block<BenchBlock> {
    // Enough ports to make it non‑trivial – add more if you like
    PortIn<float, RequiredSamples<1UZ, gr::undefined_size>> in0;
    PortIn<float, RequiredSamples<64UZ, 65536UZ>>           in1;
    PortIn<float>                                           in2;
    PortIn<float>                                           in3;
    PortOut<float>                                          out0{};

    GR_MAKE_REFLECTABLE(BenchBlock, in0, in1, in2, in3, out0);

    float processOne(float in0_, float in1_, float in2_, float in3_) { return in0_ + in1_ + in2_ + in3_; } // dummy

    TestLimits getPortLimitsLegacy() {
        auto [minS, maxS, maxA, hasA] = getPortLimits(inputPorts<PortType::STREAM>(&self()));
        return {minS, maxS, maxA, hasA};
    }

    // ---- New cache-based helper ----
    void invalidateConfig() { inputStreamCache.invalidateConfig(); }

    void invalidateStatistic() { inputStreamCache.invalidateStatistic(); }

    TestLimits getPortLimitsCache(bool recomputeAvail = true) {
        if (recomputeAvail) {
            inputStreamCache.invalidateStatistic(); // only re-read avail, config stays
        }

        // Ensure numbers are up to date (lazy inside)
        const std::size_t minSync           = inputStreamCache.minSyncRequirement();
        const std::size_t maxSync           = inputStreamCache.maxSyncRequirement();
        const std::size_t maxSyncAvailable  = inputStreamCache.maxSyncAvailable();
        const bool        hasAsyncAvailable = inputStreamCache.hasASyncAvailable();

        return {minSync, maxSync, maxSyncAvailable, hasAsyncAvailable};
    }

    std::span<const std::size_t> availableSamples(bool reset = true) { return inputStreamCache.availableSamples(reset); }
};

template<typename T>
inline void black_box(const T& v) {
    asm volatile("" : : "g"(v) : "memory");
}

// ---- UT + Bench ---------------------------------------------------
inline constexpr std::size_t N_ITER = 100'000UZ;

[[maybe_unused]] inline const boost::ut::suite port_limits_bench = [] {
    BenchBlock blk;
    blk.invalidateConfig();    // optional; first access would do it anyway
    blk.invalidateStatistic(); // optional; first access would do it anyway

    "results match (legacy vs cache)"_test = [&] {
        const auto legacy = blk.getPortLimitsLegacy();
        const auto cache  = blk.getPortLimitsCache(true);
        expect(eq(legacy.minSync, cache.minSync));
        expect(eq(legacy.maxSync, cache.maxSync));
        expect(eq(legacy.maxAvail, cache.maxAvail));
        expect(eq(legacy.hasAsync, cache.hasAsync));
    };

    // Benchmark legacy vs new
    "getPortLimits (legacy)"_benchmark.repeat<N_ITER>() = [&] {
        TestLimits r = blk.getPortLimitsLegacy();
        black_box(r);
    };

    "masked path (full compute w/ port limits) - reset all"_benchmark.repeat<N_ITER>() = [&] {
        blk.invalidateConfig();
        TestLimits r = blk.getPortLimitsCache(true);
        black_box(r);
    };

    "masked path (only available w/o port limits) - explicit reset"_benchmark.repeat<N_ITER>() = [&] {
        blk.invalidateStatistic();
        TestLimits r = blk.getPortLimitsCache(false);
        black_box(r);
    };
    "masked path (only available w/o port limits) - implicit reset"_benchmark.repeat<N_ITER>() = [&] {
        TestLimits r = blk.getPortLimitsCache(true);
        black_box(r);
    };

    "masked path (only available w/o port limits) - caching"_benchmark.repeat<N_ITER>() = [&] {
        TestLimits r = blk.getPortLimitsCache(false);
        black_box(r);
    };

    "masked path - available samples"_benchmark.repeat<N_ITER>() = [&] {
        std::span<const std::size_t> r = blk.availableSamples(true);
        black_box(r);
    };

    "masked path - available samples (cached)"_benchmark.repeat<N_ITER>() = [&] {
        std::span<const std::size_t> r = blk.availableSamples(false);
        black_box(r);
    };
};

int main() { /* tests/bench run via static init */ }
