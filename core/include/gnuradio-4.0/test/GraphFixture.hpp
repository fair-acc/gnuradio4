#ifndef GNURADIO_TEST_GRAPH_FIXTURE_HPP
#define GNURADIO_TEST_GRAPH_FIXTURE_HPP

#include <expected>
#include <utility>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

namespace gr::testing {

/**
 * A graph and the scheduler that runs it, kept together for the lifetime of a test.
 *
 * Starting a scheduler hands it the graph, and a block reference taken beforehand stays valid only as long as the
 * scheduler does. A scheduler scoped to a helper function therefore takes every block down with it on return, and
 * assertions made afterwards read freed memory — which reports as wrong values rather than as a crash.
 *
 * @code
 * gr::testing::GraphFixture fixture;
 * auto& source = fixture.emplace<MySource>({{"n", 32U}});
 * auto& sink   = fixture.emplace<MySink>();
 * expect(fixture.connect<"out", "in">(source, sink).has_value());
 * expect(fixture.run().has_value());
 * expect(eq(sink.count, 32UZ)); // sink is still alive here
 * @endcode
 */
template<typename TScheduler = gr::scheduler::Simple<>>
class GraphFixture {
public:
    gr::Graph graph;

    template<typename TBlock, typename... TArgs>
    [[nodiscard]] TBlock& emplace(TArgs&&... args) {
        return graph.emplaceBlock<TBlock>(std::forward<TArgs>(args)...);
    }

    template<typename TBlock> // a braced settings list cannot deduce through the variadic above
    [[nodiscard]] TBlock& emplace(gr::property_map settings) {
        return graph.emplaceBlock<TBlock>(std::move(settings));
    }

    template<gr::meta::fixed_string sourcePort, gr::meta::fixed_string destinationPort, typename TSource, typename TDestination>
    [[nodiscard]] auto connect(TSource& source, TDestination& destination) {
        return graph.connect<sourcePort, destinationPort>(source, destination);
    }

    [[nodiscard]] std::expected<void, gr::Error> run() {
        if (auto exchanged = _scheduler.exchange(std::move(graph)); !exchanged) {
            return std::unexpected(exchanged.error());
        }
        return _scheduler.runAndWait();
    }

    [[nodiscard]] TScheduler& scheduler() noexcept { return _scheduler; }

private:
    TScheduler _scheduler;
};

} // namespace gr::testing

#endif // GNURADIO_TEST_GRAPH_FIXTURE_HPP
