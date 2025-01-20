#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

const boost::ut::suite GraphTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "Graph connection buffer size test - default"_test = [] {
        Graph graph;
        auto& src  = graph.emplaceBlock<NullSource<float>>();
        auto& sink = graph.emplaceBlock<NullSink<float>>();

        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(src).to<"in">(sink)));
        graph.connectPendingEdges();

        expect(eq(src.out.bufferSize(), graph::defaultMinBufferSize));
        expect(eq(sink.in.bufferSize(), graph::defaultMinBufferSize));
    };

    "Graph connection buffer size test - set, one"_test = [] {
        Graph graph;
        auto& src  = graph.emplaceBlock<NullSource<float>>();
        auto& sink = graph.emplaceBlock<NullSink<float>>();

        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(src, 8000UZ).to<"in">(sink)));
        graph.connectPendingEdges();

        // Note: the actual size is always power of 2 and aligned with page size, see std::bit_ceil
        expect(eq(src.out.bufferSize(), 8192UZ));
        expect(eq(sink.in.bufferSize(), 8192UZ));
    };

    "Graph connection buffer size test - set, many"_test = [] {
        Graph graph;
        auto& src   = graph.emplaceBlock<NullSource<float>>();
        auto& sink1 = graph.emplaceBlock<NullSink<float>>();
        auto& sink2 = graph.emplaceBlock<NullSink<float>>();
        auto& sink3 = graph.emplaceBlock<NullSink<float>>();

        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(src, 2000UZ).to<"in">(sink1)));
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(src, 10000UZ).to<"in">(sink2)));
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(src, 8000UZ).to<"in">(sink3)));

        graph.connectPendingEdges();

        // Note: the actual size is always power of 2 and aligned with page size, see std::bit_ceil
        expect(eq(src.out.bufferSize(), 16384UZ));
        expect(eq(sink1.in.bufferSize(), 16384UZ));
        expect(eq(sink2.in.bufferSize(), 16384UZ));
        expect(eq(sink3.in.bufferSize(), 16384UZ));
    };
};

int main() { /* not needed for UT */ }
