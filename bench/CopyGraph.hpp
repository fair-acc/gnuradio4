#ifndef GNURADIO_BENCH_COPYGRAPH_HPP
#define GNURADIO_BENCH_COPYGRAPH_HPP

#include <tuple>
#include <utility>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

namespace gr::testing {

/** the graph every block benchmark times against a plain copy */
template<typename TBlock, typename T, typename TOut = T>
void runCopyGraph(gr::Size_t nSamples, gr::property_map settings = {}) {
    gr::Graph graph;
    auto&     source = graph.emplaceBlock<CountingSource<T>>({{"n_samples_max", nSamples}});
    auto&     block  = graph.emplaceBlock<TBlock>(std::move(settings));
    auto&     sink   = graph.emplaceBlock<CountingSink<TOut>>();
    std::ignore      = graph.connect<"out", "in">(source, block);
    std::ignore      = graph.connect<"out", "in">(block, sink);
    gr::scheduler::Simple<> scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    std::ignore = scheduler.runAndWait();
}

} // namespace gr::testing

#endif // GNURADIO_BENCH_COPYGRAPH_HPP
