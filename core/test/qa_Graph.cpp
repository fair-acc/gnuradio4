#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

template<typename T, std::size_t nPorts>
requires(std::is_arithmetic_v<T>)
struct MultiPortTestSource : public gr::Block<MultiPortTestSource<T, nPorts>> {
    std::vector<gr::PortOut<T>> out{nPorts};

    gr::Size_t              n_samples_max{1024}; // if 0 -> infinite samples
    std::vector<gr::Size_t> active_indices = {0};

    gr::Size_t _processBulkCount{0UZ};
    gr::Size_t _nSamplesProduced{0UZ};

    GR_MAKE_REFLECTABLE(MultiPortTestSource, out, n_samples_max, active_indices);

    template<gr::OutputSpanLike TOutSpan>
    gr::work::Status processBulk(std::span<TOutSpan>& outs) {
        if (active_indices.empty()) {
            std::println(std::cerr, "MultiPortTestSource::processBulk active_indices is empty");
        }

        std::size_t nSamples = 0UZ;
        for (std::size_t i = 0; i < outs.size(); i++) {
            if (std::ranges::find(active_indices, i) != active_indices.end()) {
                nSamples = outs[i].size(); // output size is the same for all ports
                outs[i].publish(nSamples);
            } else {
                outs[i].publish(0UZ);
            }
        }
        _processBulkCount++;
        _nSamplesProduced += static_cast<gr::Size_t>(nSamples);
        return _nSamplesProduced >= n_samples_max ? gr::work::Status::DONE : gr::work::Status::OK;
    }
};

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

        expect(eq(src.out.bufferSize(), graph::defaultMinBufferSize(true)));
        expect(eq(sink.in.bufferSize(), graph::defaultMinBufferSize(true)));
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

    "Graph connection buffer size test - Multi output ports"_test = [] {
        Graph graph;

        const std::size_t       customBufferSize = 8192UZ;
        const std::size_t       nIterations      = 10;
        gr::Size_t              nMaxSamples      = static_cast<gr::Size_t>(nIterations * customBufferSize);
        std::vector<gr::Size_t> activeIndices    = {0};
        auto&                   src              = graph.emplaceBlock<MultiPortTestSource<float, 3>>({{"n_samples_max", nMaxSamples}, {"active_indices", activeIndices}});
        auto&                   sink1            = graph.emplaceBlock<NullSink<float>>();

        // only the first port is connected
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out", 0>(src, customBufferSize).to<"in">(sink1)));

        scheduler::Simple<scheduler::ExecutionPolicy::multiThreaded> sched{std::move(graph)};
        expect(sched.runAndWait().has_value());

        expect(eq(src.out[0].bufferSize(), customBufferSize));
        expect(eq(sink1.in.bufferSize(), customBufferSize));
        expect(eq(src._nSamplesProduced, nMaxSamples));
        expect(eq(src._processBulkCount, 20UZ)); // it is 20 and not 10 because the not connected buffers are also included in calculation of ports limit

        expect(eq(src.out[1].bufferSize(), 4096UZ)); // port default buffer size
        expect(eq(src.out[2].bufferSize(), 4096UZ)); // port default buffer size
    };
};

int main() { /* not needed for UT */ }
