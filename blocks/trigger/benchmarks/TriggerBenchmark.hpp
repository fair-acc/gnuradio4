#ifndef GNURADIO_TRIGGER_BENCHMARK_HPP
#define GNURADIO_TRIGGER_BENCHMARK_HPP

#include <array>
#include <cmath>
#include <numbers>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

namespace gr::trigger_benchmark {

inline constexpr gr::Size_t  kSamples = 1U << 20U;
inline constexpr std::size_t kPeriod  = 64UZ;

/** a unit sine of `kPeriod` samples, repeated: two threshold crossings per period, so the work per edge is what is measured */
struct SineSource : gr::Block<SineSource> {
    gr::PortOut<float> out;
    gr::Size_t         n_samples_max = kSamples;

    GR_MAKE_REFLECTABLE(SineSource, out, n_samples_max);

    std::array<float, kPeriod> _table = [] {
        std::array<float, kPeriod> table{};
        for (std::size_t i = 0UZ; i < kPeriod; ++i) {
            table[i] = static_cast<float>(std::sin(2. * std::numbers::pi * static_cast<double>(i) / static_cast<double>(kPeriod)));
        }
        return table;
    }();
    std::size_t _produced = 0UZ;

    gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        const std::size_t n = std::min(outSpan.size(), static_cast<std::size_t>(n_samples_max) - _produced);
        for (std::size_t i = 0UZ; i < n; ++i) {
            outSpan[i] = _table[(_produced + i) % kPeriod];
        }
        _produced += n;
        outSpan.publish(n);
        return _produced == static_cast<std::size_t>(n_samples_max) ? gr::work::Status::DONE : gr::work::Status::OK;
    }
};

template<typename TBlock>
void runThrough(gr::property_map settings) {
    gr::Graph graph;
    auto&     source = graph.emplaceBlock<SineSource>();
    auto&     block  = graph.emplaceBlock<TBlock>(std::move(settings));
    auto&     sink   = graph.emplaceBlock<gr::testing::CountingSink<float>>();
    std::ignore      = graph.connect<"out", "in">(source, block);
    std::ignore      = graph.connect<"out", "in">(block, sink);
    gr::scheduler::Simple<> scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    std::ignore = scheduler.runAndWait();
}

} // namespace gr::trigger_benchmark

#endif // GNURADIO_TRIGGER_BENCHMARK_HPP
