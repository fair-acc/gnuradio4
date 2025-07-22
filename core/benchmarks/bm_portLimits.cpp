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

// #define DEFAULT_ALLOCATORS 1
#ifdef DEFAULT_ALLOCATORS
    std::vector<port::BitMask> _portTypes;
    std::vector<std::size_t>   _availableInputSamples;
    std::vector<std::size_t>   _PortMinSamples;
    std::vector<std::size_t>   _PortMaxSamples;
#else
    std::vector<port::BitMask, gr::allocator::Aligned<port::BitMask>> _portTypes;
    std::vector<std::size_t, gr::allocator::Aligned<std::size_t>>     _availableInputSamples;
    std::vector<std::size_t, gr::allocator::Aligned<std::size_t>>     _PortMinSamples;
    std::vector<std::size_t, gr::allocator::Aligned<std::size_t>>     _PortMaxSamples;
#endif

    float processOne(float in0_, float in1_, float in2_, float in3_) { return in0_ + in1_ + in2_ + in3_; } // dummy

    TestLimits getPortLimitsPublic() {
        auto [minS, maxS, maxA, hasA] = getPortLimits(inputPorts<PortType::STREAM>(&self()));
        return {minS, maxS, maxA, hasA};
    }

    void initBuffer(bool reset = true) {
        if (!reset) {
            return;
        }
        auto portTypes = getPortTypes<PortDirection::INPUT, PortType::STREAM>(self(), _portTypes);
        _availableInputSamples.resize(portTypes.size(), 0UZ);
        _PortMinSamples.resize(portTypes.size(), 0UZ);
        _PortMaxSamples.resize(portTypes.size(), gr::undefined_size);
    }

    std::span<const std::size_t> minsSpan;
    std::span<const std::size_t> maxsSpan;
    std::size_t                  minSync2;
    std::size_t                  maxSync2;

    TestLimits getPortLimitsPublicNew(bool recompute = true) noexcept {
        std::span<const std::size_t> availSpan         = getPortConstraints<PortDirection::INPUT, PortType::STREAM>(self(), _availableInputSamples, [](auto& port) { return port.streamReader().available(); });
        const std::size_t            maxSyncAvailable  = min_element_masked<PortSync::SYNCHRONOUS>(availSpan, _portTypes).value_or(gr::undefined_size);
        const bool                   hasAsyncAvailable = compareRangesMasked<PortSync::ASYNCHRONOUS>(availSpan, minsSpan, _portTypes);

        if (recompute || minsSpan.empty() || maxsSpan.empty()) [[unlikely]] {
            minsSpan = getPortConstraints<PortDirection::INPUT, PortType::STREAM>(self(), _PortMinSamples, [](auto& port) { return port.min_samples; });
            maxsSpan = getPortConstraints<PortDirection::INPUT, PortType::STREAM>(self(), _PortMaxSamples, [](auto& port) { return port.max_samples; });
            minSync2 = max_element_masked<PortSync::SYNCHRONOUS>(minsSpan, _portTypes).value_or(0UZ);
            maxSync2 = min_element_masked<PortSync::SYNCHRONOUS>(maxsSpan, _portTypes).value_or(gr::undefined_size);
        }

        return {minSync2, maxSync2, maxSyncAvailable, hasAsyncAvailable};
    }
};

template<typename T>
inline void black_box(const T& v) {
    asm volatile("" : : "g"(v) : "memory");
}

// ---- UT + Bench ---------------------------------------------------
inline constexpr std::size_t N_ITER    = 100UZ;
inline constexpr std::size_t N_SAMPLES = 1'000'000'000UZ;

[[maybe_unused]] inline const boost::ut::suite port_limits_bench = [] {
    BenchBlock blk;
    blk.initBuffer();

    // Snapshot once to feed both impls the same state (deterministic UT)
    // BUT the benchmark below uses “live” calls (that’s what you asked to time)
    "test_result"_test = [&] {
        const auto l = blk.getPortLimitsPublic();
        const auto n = blk.getPortLimitsPublicNew();
        expect(eq(l.minSync, n.minSync));
        expect(eq(l.maxSync, n.maxSync));
        expect(eq(l.maxAvail, n.maxAvail));
        expect(eq(l.hasAsync, n.hasAsync));
    };

    // Benchmark legacy vs new
    "getPortLimits (legacy)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&] {
        auto r = blk.getPortLimitsPublic();
        black_box(r);
    };

    "masked path (full compute w/ port limits)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&] {
        auto r = blk.getPortLimitsPublicNew();
        black_box(r);
    };

    "masked path (only available w/o port limits)"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&] {
        auto r = blk.getPortLimitsPublicNew(false);
        black_box(r);
    };
};

int main() { /* tests run statically */ }
