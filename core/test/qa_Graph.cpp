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

const boost::ut::suite<"GraphTests"> _1 = [] {
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

        scheduler::Simple<scheduler::ExecutionPolicy::multiThreaded> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());

        expect(eq(src.out[0].bufferSize(), customBufferSize));
        expect(eq(sink1.in.bufferSize(), customBufferSize));
        expect(eq(src._nSamplesProduced, nMaxSamples));
        expect(eq(src._processBulkCount, 20UZ)); // it is 20 and not 10 because the not connected buffers are also included in calculation of ports limit

        expect(eq(src.out[1].bufferSize(), 4096UZ)); // port default buffer size
        expect(eq(src.out[2].bufferSize(), 4096UZ)); // port default buffer size
    };
};

const boost::ut::suite<"GraphExtensionsTests"> _2 = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "findBlock by name"_test = [] {
        Graph                               graph;
        [[maybe_unused]] NullSource<float>& src = graph.emplaceBlock<NullSource<float>>();
        NullSink<float>&                    snk = graph.emplaceBlock<NullSink<float>>();

        std::expected<std::shared_ptr<BlockModel>, Error> findSinkBlock = graph::findBlock(graph, snk.unique_name);
        expect(findSinkBlock.has_value());
        expect(eq(findSinkBlock.value()->uniqueName(), snk.unique_name));

        expect(!graph::findBlock(graph, "bogus").has_value());
    };

    "findBlock by BlockLike&"_test = [] {
        Graph                                             graph;
        NullSource<float>&                                block  = graph.emplaceBlock<NullSource<float>>();
        std::expected<std::shared_ptr<BlockModel>, Error> result = graph::findBlock(graph, block);
        expect(result.has_value());
        expect(eq(result.value()->uniqueName(), block.unique_name));

        NullSource<float> other;
        expect(!graph::findBlock(graph, other).has_value());
    };

    "findBlock by shared_ptr<BlockModel>"_test = [] {
        Graph                                             graph;
        NullSource<float>&                                block    = graph.emplaceBlock<NullSource<float>>();
        std::shared_ptr<BlockModel>                       blockPtr = graph::findBlock(graph, block).value();
        std::expected<std::shared_ptr<BlockModel>, Error> result   = graph::findBlock(graph, blockPtr);
        expect(result.has_value());
        expect(eq(result.value()->uniqueName(), block.unique_name));

        std::shared_ptr<BlockModel> bogus = std::make_shared<BlockWrapper<NullSource<float>>>();
        expect(!graph::findBlock(graph, bogus).has_value());
    };

    "containsEdge returns true after connection"_test = [] {
        Graph              graph;
        NullSource<float>& src = graph.emplaceBlock<NullSource<float>>();
        NullSink<float>&   snk = graph.emplaceBlock<NullSink<float>>();
        expect(eq(graph.connect<"out">(src).to<"in">(snk), ConnectionResult::SUCCESS));

        expect(graph.containsEdge(graph.edges().front()));
        graph.connectPendingEdges();
        expect(graph.containsEdge(graph.edges().front()));
    };

    "addEdge and removeEdge work correctly"_test = [] {
        Graph              graph;
        NullSource<float>& src = graph.emplaceBlock<NullSource<float>>();
        NullSink<float>&   snk = graph.emplaceBlock<NullSink<float>>();
        expect(eq(graph.connect<"out">(src).to<"in">(snk), ConnectionResult::SUCCESS));
        graph.connectPendingEdges();

        const auto edge = graph.edges().front();
        expect(graph.containsEdge(edge));
        expect(graph.removeEdge(edge));
        expect(!graph.containsEdge(edge));
    };

    "forEachBlock visits all blocks"_test = [] {
        Graph                    graph;
        std::vector<std::string> visited;

        NullSource<float>& src = graph.emplaceBlock<NullSource<float>>();
        NullSink<float>&   snk = graph.emplaceBlock<NullSink<float>>();

        graph::forEachBlock<gr::block::Category::TransparentBlockGroup>(graph, [&](std::shared_ptr<BlockModel> block) { //
            visited.push_back(std::string(block->uniqueName()));
        });

        expect(eq(visited.size(), 2UZ));
        expect(std::ranges::find(visited, src.unique_name) != visited.end());
        expect(std::ranges::find(visited, snk.unique_name) != visited.end());
    };

    "forEachEdge visits all edges"_test = [] {
        Graph              graph;
        NullSource<float>& src = graph.emplaceBlock<NullSource<float>>();
        NullSink<float>&   snk = graph.emplaceBlock<NullSink<float>>();

        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(src).to<"in">(snk)));
        graph.connectPendingEdges();

        int count = 0;
        graph::forEachEdge<gr::block::Category::TransparentBlockGroup>(graph, [&](auto) { ++count; });
        expect(eq(count, 1));
    };

    "traverseSubgraphs visits nested blocks"_test = [] {
        std::shared_ptr<BlockModel> wrappedGraph = std::make_shared<BlockWrapper<Graph>>();
        Graph*                      root         = static_cast<Graph*>(wrappedGraph->raw());

        auto& src    = root->emplaceBlock<NullSource<float>>();
        auto& nested = root->emplaceBlock<Graph>();
        auto& sink   = nested.emplaceBlock<NullSink<float>>();

        using enum gr::block::Category;
        "visit transparend (unmanaged) sub-graphs"_test = [&] {
            std::vector<std::string> visited;
            gr::graph::detail::traverseSubgraphs<TransparentBlockGroup>(*wrappedGraph, [&](auto& graph) {
                for (const auto& block : graph.blocks()) {
                    visited.push_back(std::string(block->uniqueName()));
                }
            });

            expect(eq(visited.size(), 3UZ)) << std::format("visited:\n{}\n", gr::join(visited, "\n"));
            expect(std::ranges::find(visited, src.unique_name) != visited.end()) << std::format("couldn't find '{}' in '{}", src.unique_name, gr::join(visited, ", "));
            expect(std::ranges::find(visited, nested.unique_name) != visited.end()) << std::format("couldn't find '{}' in '{}", nested.unique_name, gr::join(visited, ", "));
            expect(std::ranges::find(visited, sink.unique_name) != visited.end()) << std::format("couldn't find '{}' in '{}", sink.unique_name, gr::join(visited, ", "));
        };

        "visit nmanaged sub-graphs"_test = [&] {
            std::vector<std::string> visited;
            gr::graph::detail::traverseSubgraphs<ScheduledBlockGroup>(*wrappedGraph, [&](auto& graph) {
                for (const auto& block : graph.blocks()) {
                    visited.push_back(std::string(block->uniqueName()));
                }
            });

            expect(eq(visited.size(), 2UZ)) << std::format("visited:\n{}\n", gr::join(visited, "\n"));
            expect(std::ranges::find(visited, src.unique_name) != visited.end()) << std::format("couldn't find '{}' in '{}", src.unique_name, gr::join(visited, ", "));
            expect(std::ranges::find(visited, nested.unique_name) != visited.end()) << std::format("couldn't find '{}' in '{}", nested.unique_name, gr::join(visited, ", ")); // in because it acts like a block
            expect(std::ranges::find(visited, sink.unique_name) == visited.end()) << std::format("couldn't find '{}' in '{}", sink.unique_name, gr::join(visited, ", "));
        };

        "visit all sub-graphs"_test = [&] {
            std::vector<std::string> visited;
            gr::graph::detail::traverseSubgraphs<All>(*wrappedGraph, [&](auto& graph) {
                for (const auto& block : graph.blocks()) {
                    visited.push_back(std::string(block->uniqueName()));
                }
            });

            expect(eq(visited.size(), 3UZ)) << std::format("visited:\n{}\n", gr::join(visited, "\n"));
            expect(std::ranges::find(visited, src.unique_name) != visited.end()) << std::format("couldn't find '{}' in '{}", src.unique_name, gr::join(visited, ", "));
            expect(std::ranges::find(visited, nested.unique_name) != visited.end()) << std::format("couldn't find '{}' in '{}", nested.unique_name, gr::join(visited, ", "));
            expect(std::ranges::find(visited, sink.unique_name) != visited.end()) << std::format("couldn't find '{}' in '{}", sink.unique_name, gr::join(visited, ", "));
        };

        "visit top-level Blocks only"_test = [&] {
            std::vector<std::string> visited;
            gr::graph::detail::traverseSubgraphs<NormalBlock>(*wrappedGraph, [&](auto& graph) {
                for (const auto& block : graph.blocks()) {
                    visited.push_back(std::string(block->uniqueName()));
                }
            });

            expect(eq(visited.size(), 2UZ)) << std::format("visited:\n{}\n", gr::join(visited, "\n"));
            expect(std::ranges::find(visited, src.unique_name) != visited.end()) << std::format("couldn't find '{}' in '{}", src.unique_name, gr::join(visited, ", "));
            expect(std::ranges::find(visited, nested.unique_name) != visited.end()) << std::format("couldn't find '{}' in '{}", nested.unique_name, gr::join(visited, ", ")); // in because it acts like a block
            expect(std::ranges::find(visited, sink.unique_name) == visited.end()) << std::format("couldn't find '{}' in '{}", sink.unique_name, gr::join(visited, ", "));
        };
    };
};

int main() { /* not needed for UT */ }
