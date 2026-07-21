#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <optional>

template<typename T, std::size_t nPorts>
requires(std::is_arithmetic_v<T>)
struct MultiPortTestSource : public gr::Block<MultiPortTestSource<T, nPorts>> {
    std::vector<gr::PortOut<T>> out{nPorts};

    gr::Size_t             n_samples_max{1024}; // if 0 -> infinite samples
    gr::Tensor<gr::Size_t> active_indices = {gr::data_from, {0}};

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

const boost::ut::suite<"New connection API tests"> connection_api_tests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "Graph connection buffer size test - default"_test = [] {
        Graph graph;
        auto& src  = graph.emplaceBlock<NullSource<float>>();
        auto& sink = graph.emplaceBlock<NullSink<float>>();

        expect(graph.connect<"out", "in">(src, sink).has_value());
        graph.connectPendingEdges();

        expect(eq(src.out.bufferSize(), graph::defaultMinBufferSize(true)));
        expect(eq(sink.in.bufferSize(), graph::defaultMinBufferSize(true)));
    };
};

const boost::ut::suite<"GraphTests"> _1 = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "Graph connection buffer size test - default"_test = [] {
        Graph graph;
        auto& src  = graph.emplaceBlock<NullSource<float>>();
        auto& sink = graph.emplaceBlock<NullSink<float>>();

        expect(graph.connect<"out", "in">(src, sink, {.minBufferSize = undefined_size}).has_value());
        graph.connectPendingEdges();

        expect(eq(src.out.bufferSize(), graph::defaultMinBufferSize(true)));
        expect(eq(sink.in.bufferSize(), graph::defaultMinBufferSize(true)));
    };

    "Graph connection buffer size test - set, one"_test = [] {
        Graph graph;
        auto& src  = graph.emplaceBlock<NullSource<float>>();
        auto& sink = graph.emplaceBlock<NullSink<float>>();

        expect(graph.connect<"out", "in">(src, sink, {.minBufferSize = 8000UZ}).has_value());
        graph.connectPendingEdges();

        expect(ge(src.out.bufferSize(), 8000UZ));
        expect(ge(sink.in.bufferSize(), 8000UZ));
    };

    "Graph connection buffer size test - set, many"_test = [] {
        Graph graph;
        auto& src   = graph.emplaceBlock<NullSource<float>>();
        auto& sink1 = graph.emplaceBlock<NullSink<float>>();
        auto& sink2 = graph.emplaceBlock<NullSink<float>>();
        auto& sink3 = graph.emplaceBlock<NullSink<float>>();

        expect(graph.connect<"out", "in">(src, sink1, {.minBufferSize = 2000UZ}).has_value());
        expect(graph.connect<"out", "in">(src, sink2, {.minBufferSize = 10000UZ}).has_value());
        expect(graph.connect<"out", "in">(src, sink3, {.minBufferSize = 8000UZ}).has_value());

        graph.connectPendingEdges();

        // contract: min buffer is at least as larges 'minBufferSize' connection requirement
        const std::size_t maxBuffer = std::max<std::size_t>(2000UZ, std::max<std::size_t>(10000UZ, 8000UZ));
        expect(ge(src.out.bufferSize(), maxBuffer));
        expect(ge(sink1.in.bufferSize(), maxBuffer));
        expect(ge(sink2.in.bufferSize(), maxBuffer));
        expect(ge(sink3.in.bufferSize(), maxBuffer));
    };

    "Graph connection buffer size test - Multi output ports"_test = [] {
        Graph graph;

        const std::size_t  customBufferSize = 8192UZ;
        const std::size_t  nIterations      = 10;
        gr::Size_t         nMaxSamples      = static_cast<gr::Size_t>(nIterations * customBufferSize);
        Tensor<gr::Size_t> activeIndices    = {gr::data_from, {0}};
        auto&              src              = graph.emplaceBlock<MultiPortTestSource<float, 3>>({{"n_samples_max", nMaxSamples}, {"active_indices", activeIndices}});
        auto&              sink1            = graph.emplaceBlock<NullSink<float>>();

        // only the first port is connected
        expect(graph.connect(src, "out#0", sink1, "in", {.minBufferSize = customBufferSize}).has_value());

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

struct TrackingResource : std::pmr::memory_resource {
    std::pmr::memory_resource* _upstream;
    std::atomic<std::size_t>   _allocCount{0};

    explicit TrackingResource(std::pmr::memory_resource* upstream = std::pmr::get_default_resource()) : _upstream(upstream) {}

    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        ++_allocCount;
        return _upstream->allocate(bytes, alignment);
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override { _upstream->deallocate(p, bytes, alignment); }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
};

const boost::ut::suite<"EdgeParameters PMR forwarding"> _pmr = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "Graph connection forwards data PMR resource"_test = [] {
        TrackingResource dataTracker;
        Graph            graph;
        auto&            src  = graph.emplaceBlock<NullSource<float>>();
        auto&            sink = graph.emplaceBlock<NullSink<float>>();

        expect(graph.connect<"out", "in">(src, sink, {.minBufferSize = 4096UZ, .dataResource = &dataTracker}).has_value());
        graph.connectPendingEdges();

        expect(gt(dataTracker._allocCount.load(), 0UZ)) << "data PMR resource should have been used for stream buffer";
    };

    "Graph connection forwards tag PMR resource"_test = [] {
        TrackingResource tagTracker;
        Graph            graph;
        auto&            src  = graph.emplaceBlock<NullSource<float>>();
        auto&            sink = graph.emplaceBlock<NullSink<float>>();

        expect(graph.connect<"out", "in">(src, sink, {.minBufferSize = 4096UZ, .tagResource = &tagTracker}).has_value());
        graph.connectPendingEdges();

        expect(gt(tagTracker._allocCount.load(), 0UZ)) << "tag PMR resource should have been used for tag buffer";
    };

    "Graph connection forwards both PMR resources"_test = [] {
        TrackingResource dataTracker;
        TrackingResource tagTracker;
        Graph            graph;
        auto&            src  = graph.emplaceBlock<NullSource<float>>();
        auto&            sink = graph.emplaceBlock<NullSink<float>>();

        expect(graph.connect<"out", "in">(src, sink, {.minBufferSize = 4096UZ, .dataResource = &dataTracker, .tagResource = &tagTracker}).has_value());
        graph.connectPendingEdges();

        expect(gt(dataTracker._allocCount.load(), 0UZ)) << "data PMR resource should have been used";
        expect(gt(tagTracker._allocCount.load(), 0UZ)) << "tag PMR resource should have been used";
    };

    "Graph connection with default PMR resources still works"_test = [] {
        Graph graph;
        auto& src  = graph.emplaceBlock<NullSource<float>>();
        auto& sink = graph.emplaceBlock<NullSink<float>>();

        expect(graph.connect<"out", "in">(src, sink, {.minBufferSize = 4096UZ}).has_value());
        graph.connectPendingEdges();

        expect(ge(src.out.bufferSize(), 4096UZ));
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

    "blockIndex by name"_test = [] {
        Graph              graph;
        NullSource<float>& block1 = graph.emplaceBlock<NullSource<float>>();
        NullSource<float>& block2 = graph.emplaceBlock<NullSource<float>>();
        expect(eq(graph::blockIndex(graph, block1.unique_name).value(), 0UZ));
        expect(eq(graph::blockIndex(graph, block2.unique_name).value(), 1UZ));
        expect(!graph::blockIndex(graph, "unknownBlock").has_value());
    };

    "blockIndex by shared_ptr<BlockModel>"_test = [] {
        Graph                       graph;
        NullSource<float>&          block1    = graph.emplaceBlock<NullSource<float>>();
        NullSource<float>&          block2    = graph.emplaceBlock<NullSource<float>>();
        std::shared_ptr<BlockModel> blockPtr1 = graph::findBlock(graph, block1).value();
        std::shared_ptr<BlockModel> blockPtr2 = graph::findBlock(graph, block2).value();
        expect(eq(graph::blockIndex(graph, blockPtr1).value(), 0UZ));
        expect(eq(graph::blockIndex(graph, blockPtr2).value(), 1UZ));

        std::shared_ptr<BlockModel> bogus = std::make_shared<BlockWrapper<NullSource<float>>>();
        expect(!graph::blockIndex(graph, bogus).has_value());
    };

    "containsEdge returns true after connection"_test = [] {
        Graph              graph;
        NullSource<float>& src = graph.emplaceBlock<NullSource<float>>();
        NullSink<float>&   snk = graph.emplaceBlock<NullSink<float>>();
        expect(graph.connect<"out", "in">(src, snk).has_value());

        expect(graph.containsEdge(graph.edges().front()));
        graph.connectPendingEdges();
        expect(graph.containsEdge(graph.edges().front()));
    };

    "addEdge and removeEdge work correctly"_test = [] {
        Graph              graph;
        NullSource<float>& src = graph.emplaceBlock<NullSource<float>>();
        NullSink<float>&   snk = graph.emplaceBlock<NullSink<float>>();
        expect(graph.connect<"out", "in">(src, snk).has_value());
        graph.connectPendingEdges();

        const auto edge = graph.edges().front();
        expect(graph.containsEdge(edge));
        expect(graph.removeEdge(edge));
        expect(!graph.containsEdge(edge));
    };

    "emplaceEdge then removeEdgeBySourcePort erases the edge"_test = [] {
        Graph              graph;
        NullSource<float>& src = graph.emplaceBlock<NullSource<float>>();
        NullSink<float>&   snk = graph.emplaceBlock<NullSink<float>>();

        const auto emplaced = graph.emplaceEdge(src.unique_name.value(), "out", snk.unique_name.value(), "in", undefined_size, 0, "unnamed edge");
        expect(emplaced.has_value()) << [&] { return emplaced ? std::string{} : emplaced.error().message; } << fatal;
        expect(eq(graph.edges().size(), 1UZ));

        const auto removed = graph.removeEdgeBySourcePort(src.unique_name.value(), "out");
        expect(removed.has_value()) << [&] { return removed ? std::string{} : removed.error().message; } << fatal;
        expect(eq(graph.edges().size(), 0UZ)) << "removed edge must not remain in the graph's edge list";
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
        expect(std::ranges::find(visited, std::string(src.unique_name.value())) != visited.end());
        expect(std::ranges::find(visited, std::string(snk.unique_name.value())) != visited.end());
    };

    "forEachEdge visits all edges"_test = [] {
        Graph              graph;
        NullSource<float>& src = graph.emplaceBlock<NullSource<float>>();
        NullSink<float>&   snk = graph.emplaceBlock<NullSink<float>>();

        expect(graph.connect<"out", "in">(src, snk, {.minBufferSize = undefined_size}).has_value());
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
            expect(std::ranges::find(visited, std::string(src.unique_name.value())) != visited.end()) << std::format("couldn't find '{}' in '{}", src.unique_name, gr::join(visited, ", "));
            expect(std::ranges::find(visited, std::string(nested.unique_name.value())) != visited.end()) << std::format("couldn't find '{}' in '{}", nested.unique_name, gr::join(visited, ", "));
            expect(std::ranges::find(visited, std::string(sink.unique_name.value())) != visited.end()) << std::format("couldn't find '{}' in '{}", sink.unique_name, gr::join(visited, ", "));
        };

        "visit nmanaged sub-graphs"_test = [&] {
            std::vector<std::string> visited;
            gr::graph::detail::traverseSubgraphs<ScheduledBlockGroup>(*wrappedGraph, [&](auto& graph) {
                for (const auto& block : graph.blocks()) {
                    visited.push_back(std::string(block->uniqueName()));
                }
            });

            expect(eq(visited.size(), 2UZ)) << std::format("visited:\n{}\n", gr::join(visited, "\n"));
            expect(std::ranges::find(visited, std::string(src.unique_name.value())) != visited.end()) << std::format("couldn't find '{}' in '{}", src.unique_name, gr::join(visited, ", "));
            expect(std::ranges::find(visited, std::string(nested.unique_name.value())) != visited.end()) << std::format("couldn't find '{}' in '{}", nested.unique_name, gr::join(visited, ", ")); // in because it acts like a block
            expect(std::ranges::find(visited, std::string(sink.unique_name.value())) == visited.end()) << std::format("couldn't find '{}' in '{}", sink.unique_name, gr::join(visited, ", "));
        };

        "visit all sub-graphs"_test = [&] {
            std::vector<std::string> visited;
            gr::graph::detail::traverseSubgraphs<All>(*wrappedGraph, [&](auto& graph) {
                for (const auto& block : graph.blocks()) {
                    visited.push_back(std::string(block->uniqueName()));
                }
            });

            expect(eq(visited.size(), 3UZ)) << std::format("visited:\n{}\n", gr::join(visited, "\n"));
            expect(std::ranges::find(visited, std::string(src.unique_name.value())) != visited.end()) << std::format("couldn't find '{}' in '{}", src.unique_name, gr::join(visited, ", "));
            expect(std::ranges::find(visited, std::string(nested.unique_name.value())) != visited.end()) << std::format("couldn't find '{}' in '{}", nested.unique_name, gr::join(visited, ", "));
            expect(std::ranges::find(visited, std::string(sink.unique_name.value())) != visited.end()) << std::format("couldn't find '{}' in '{}", sink.unique_name, gr::join(visited, ", "));
        };

        "visit top-level Blocks only"_test = [&] {
            std::vector<std::string> visited;
            gr::graph::detail::traverseSubgraphs<NormalBlock>(*wrappedGraph, [&](auto& graph) {
                for (const auto& block : graph.blocks()) {
                    visited.push_back(std::string(block->uniqueName()));
                }
            });

            expect(eq(visited.size(), 2UZ)) << std::format("visited:\n{}\n", gr::join(visited, "\n"));
            expect(std::ranges::find(visited, std::string(src.unique_name.value())) != visited.end()) << std::format("couldn't find '{}' in '{}", src.unique_name, gr::join(visited, ", "));
            expect(std::ranges::find(visited, std::string(nested.unique_name.value())) != visited.end()) << std::format("couldn't find '{}' in '{}", nested.unique_name, gr::join(visited, ", ")); // in because it acts like a block
            expect(std::ranges::find(visited, std::string(sink.unique_name.value())) == visited.end()) << std::format("couldn't find '{}' in '{}", sink.unique_name, gr::join(visited, ", "));
        };
    };
};

template<gr::block::Category BlockCategory>
void visitBlocks(gr::Graph& graph, size_t nExpected, std::vector<std::string> expectedNames = {}, //
    gr::block::Category filter = gr::block::Category::All, std::source_location location = std::source_location::current()) {
    using namespace boost::ut;
    std::vector<std::string> visited;

    gr::graph::forEachBlock<BlockCategory>(
        graph,
        [&](auto& block) { //
            visited.emplace_back(block->uniqueName());
        },
        filter);

    expect(eq(visited.size(), nExpected)) << std::format("visited:\n{}\n location={}\n", gr::join(visited, "\n"), location);
    for (const auto& name : expectedNames) {
        expect(std::ranges::find(visited, name) != visited.end()) << std::format("couldn't find '{}' in '{} location={}", name, gr::join(visited, ", "), location);
    }
};

const boost::ut::suite<"forEachBlock"> _3 = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "No nesting"_test = [] {
        Graph              graph;
        NullSource<float>& src = graph.emplaceBlock<NullSource<float>>();
        NullSink<float>&   snk = graph.emplaceBlock<NullSink<float>>();

        visitBlocks<gr::block::Category::All>(graph, 2UZ, {std::string(src.unique_name.value()), std::string(snk.unique_name.value())});
        visitBlocks<gr::block::Category::TransparentBlockGroup>(graph, 2UZ, {std::string(src.unique_name.value()), std::string(snk.unique_name.value())});
        visitBlocks<gr::block::Category::NormalBlock>(graph, 2UZ, {std::string(src.unique_name.value()), std::string(snk.unique_name.value())});
        visitBlocks<gr::block::Category::ScheduledBlockGroup>(graph, 2UZ, {std::string(src.unique_name.value()), std::string(snk.unique_name.value())});
    };

    "unmanaged sub-graph"_test = [] {
        Graph root;
        Graph subGraph;
        auto& subSrc        = subGraph.emplaceBlock<NullSource<float>>();
        auto& subSnk        = subGraph.emplaceBlock<NullSink<float>>();
        auto  subGraphModel = std::unique_ptr<BlockModel>(std::make_unique<GraphWrapper<Graph>>(std::move(subGraph)).release());

        auto& src         = root.emplaceBlock<NullSource<float>>();
        auto  nestedGraph = root.addBlock(std::move(subGraphModel));
        auto& sink        = root.emplaceBlock<NullSink<float>>();

        visitBlocks<gr::block::Category::All>(root, 5UZ, {std::string(src.unique_name.value()), std::string(sink.unique_name.value()), std::string(subSrc.unique_name.value()), std::string(subSnk.unique_name.value()), std::string(nestedGraph->uniqueName())});
        visitBlocks<gr::block::Category::All>(root, 4UZ, {std::string(src.unique_name.value()), std::string(sink.unique_name.value()), std::string(subSrc.unique_name.value()), std::string(subSnk.unique_name.value())}, //
            gr::block::Category::NormalBlock);
        visitBlocks<gr::block::Category::All>(root, 1UZ, {std::string(nestedGraph->uniqueName())}, //
            gr::block::Category::TransparentBlockGroup);
        visitBlocks<gr::block::Category::All>(root, 0UZ, {}, //
            gr::block::Category::ScheduledBlockGroup);

        visitBlocks<gr::block::Category::TransparentBlockGroup>(root, 5UZ, {std::string(src.unique_name.value()), std::string(sink.unique_name.value()), std::string(subSrc.unique_name.value()), std::string(subSnk.unique_name.value()), std::string(nestedGraph->uniqueName())});
        visitBlocks<gr::block::Category::TransparentBlockGroup>(root, 4UZ, {std::string(src.unique_name.value()), std::string(sink.unique_name.value()), std::string(subSrc.unique_name.value()), std::string(subSnk.unique_name.value())}, //
            gr::block::Category::NormalBlock);
        visitBlocks<gr::block::Category::TransparentBlockGroup>(root, 1UZ, {std::string(nestedGraph->uniqueName())}, //
            gr::block::Category::TransparentBlockGroup);
        visitBlocks<gr::block::Category::TransparentBlockGroup>(root, 0UZ, {}, //
            gr::block::Category::ScheduledBlockGroup);
    };

    "managed sub-graph"_test = [] {
        using Scheduler = gr::scheduler::Simple<scheduler::ExecutionPolicy::multiThreaded>;

        Graph root;
        Graph subGraph;
        auto& subSrc = subGraph.emplaceBlock<NullSource<float>>();
        auto& subSnk = subGraph.emplaceBlock<NullSink<float>>();

        auto schedulerModel = std::unique_ptr<BlockModel>(std::make_unique<SchedulerWrapper<Scheduler>>().release());
        auto wrapper        = static_cast<SchedulerWrapper<Scheduler>*>(schedulerModel.get());
        wrapper->setGraph(std::move(subGraph));

        auto& src             = root.emplaceBlock<NullSource<float>>();
        auto  nestedScheduler = root.addBlock(std::move(schedulerModel));
        auto& sink            = root.emplaceBlock<NullSink<float>>();

        visitBlocks<gr::block::Category::All>(root, 5UZ, {std::string(src.unique_name.value()), std::string(sink.unique_name.value()), std::string(subSrc.unique_name.value()), std::string(subSnk.unique_name.value()), std::string(nestedScheduler->uniqueName())});
        visitBlocks<gr::block::Category::All>(root, 4UZ, {std::string(src.unique_name.value()), std::string(sink.unique_name.value()), std::string(subSrc.unique_name.value()), std::string(subSnk.unique_name.value())}, //
            gr::block::Category::NormalBlock);
        visitBlocks<gr::block::Category::All>(root, 1UZ, {std::string(nestedScheduler->uniqueName())}, //
            gr::block::Category::ScheduledBlockGroup);
        visitBlocks<gr::block::Category::All>(root, 0UZ, {}, //
            gr::block::Category::TransparentBlockGroup);

        expect(subSrc.blockCategory == gr::block::Category::NormalBlock) << std::format("subSrc.blockCategory = {}", static_cast<int>(subSrc.blockCategory));
        expect(subSnk.blockCategory == gr::block::Category::NormalBlock) << std::format("subSnk.blockCategory = {}", static_cast<int>(subSnk.blockCategory));
        expect(nestedScheduler->blockCategory() == gr::block::Category::ScheduledBlockGroup) << std::format("nestedScheduler->blockCategory() = {}", static_cast<int>(nestedScheduler->blockCategory()));
        expect(src.blockCategory == gr::block::Category::NormalBlock) << std::format("src.blockCategory = {}", static_cast<int>(src.blockCategory));
        expect(sink.blockCategory == gr::block::Category::NormalBlock) << std::format("sink.blockCategory = {}", static_cast<int>(sink.blockCategory));

        expect(eq(root.blocks().size(), 3UZ)) << "root.blocks().size()";
        expect(eq(nestedScheduler->graph()->blocks().size(), 2UZ)) << "nestedScheduler->graph()->blocks().size()";

        visitBlocks<gr::block::Category::TransparentBlockGroup>(root, 3UZ, {std::string(src.unique_name.value()), std::string(sink.unique_name.value()), std::string(nestedScheduler->uniqueName())});
        visitBlocks<gr::block::Category::TransparentBlockGroup>(root, 2UZ, {std::string(src.unique_name.value()), std::string(sink.unique_name.value())}, //
            gr::block::Category::NormalBlock);
        visitBlocks<gr::block::Category::TransparentBlockGroup>(root, 1UZ, {std::string(nestedScheduler->uniqueName())}, //
            gr::block::Category::ScheduledBlockGroup);
        visitBlocks<gr::block::Category::TransparentBlockGroup>(root, 0UZ, {}, gr::block::Category::TransparentBlockGroup);
    };
};

namespace {
struct TestMR : std::pmr::memory_resource {
    std::size_t allocCount = 0;
    void*       do_allocate(std::size_t n, std::size_t) override {
        ++allocCount;
        return ::operator new(n == 0 ? 1 : n);
    }
    void do_deallocate(void* p, std::size_t, std::size_t) override { ::operator delete(p); }
    bool do_is_equal(const std::pmr::memory_resource& o) const noexcept override { return this == &o; }
};

std::pmr::memory_resource* testProvider(const gr::ComputeDomain&, void* ctx) { return static_cast<std::pmr::memory_resource*>(ctx); }
} // namespace

const boost::ut::suite<"Edge domain resolution"> _edgeDomainResolution = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "edge with explicit domain resolves resource"_test = [] {
        TestMR mr;
        ComputeRegistry::instance().register_provider("test-edge", &testProvider);

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        EdgeParameters params;
        params.domain      = ComputeDomain::gpu_shared("test-edge");
        params.domain.user = &mr;
        expect(testGraph.connect<"out", "in">(src, sink, params).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        auto edges = sched.graph().edges();
        expect(eq(edges.size(), 1UZ));
        expect(edges[0]._dataResource == &mr) << "edge buffer must use resolved resource";
        expect(edges[0]._tagResource == &mr) << "tag buffer must use resolved resource";
    };

    "edge with host domain uses default resource"_test = [] {
        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        auto edges = sched.graph().edges();
        expect(eq(edges.size(), 1UZ));
        expect(edges[0]._dataResource == std::pmr::get_default_resource()) << "host domain must use default resource";
    };

    "explicit dataResource overrides domain resolution"_test = [] {
        TestMR explicitMr;
        ComputeRegistry::instance().register_provider("test-override", &testProvider);

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        EdgeParameters params;
        params.domain       = ComputeDomain::gpu_shared("test-override");
        params.dataResource = &explicitMr;
        params.tagResource  = &explicitMr;
        expect(testGraph.connect<"out", "in">(src, sink, params).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        auto edges = sched.graph().edges();
        expect(eq(edges.size(), 1UZ));
        expect(edges[0]._dataResource == &explicitMr) << "explicit resource must override domain resolution";
    };

    "block compute_domain auto-resolves edge resource"_test = [] {
        TestMR mr;
        ComputeRegistry::instance().register_provider("test-auto", &testProvider);

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}, {"compute_domain", "gpu:test-auto"}});
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        // connect without explicit EdgeParameters — domain should auto-resolve from block compute_domain
        expect(testGraph.connect<"out", "in">(src, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        auto edges = sched.graph().edges();
        expect(eq(edges.size(), 1UZ));
        expect(eq(edges[0]._domain.kind, "gpu"sv)) << "domain kind auto-resolved from block compute_domain";
        expect(eq(edges[0]._domain.backend, "test-auto"sv)) << "domain backend auto-resolved";
    };

    "explicit EdgeParameters.domain overrides block compute_domain"_test = [] {
        TestMR mr;
        ComputeRegistry::instance().register_provider("test-explicit-dom", &testProvider);

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}, {"compute_domain", "gpu:test-auto"}});
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        EdgeParameters params;
        params.domain      = ComputeDomain::gpu_shared("test-explicit-dom");
        params.domain.user = &mr;
        expect(testGraph.connect<"out", "in">(src, sink, params).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        auto edges = sched.graph().edges();
        expect(eq(edges.size(), 1UZ));
        expect(edges[0]._dataResource == &mr) << "explicit EdgeParameters.domain must override block compute_domain";
        expect(eq(edges[0]._domain.backend, "test-explicit-dom"sv)) << "explicit domain backend preserved";
    };

    "two CPU blocks produce default edge"_test = [] {
        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(10)}, {"verbose_console", false}});
        auto& sink = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"verbose_console", false}});

        expect(testGraph.connect<"out", "in">(src, sink).has_value());

        scheduler::Simple<> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());
        expect(sched.runAndWait().has_value());

        auto edges = sched.graph().edges();
        expect(eq(edges.size(), 1UZ));
        expect(eq(edges[0]._domain.kind, "host"sv)) << "default CPU blocks must produce host domain edge";
        expect(edges[0]._dataResource == std::pmr::get_default_resource()) << "default resource";
    };
};

const boost::ut::suite<"edge PMR resource precedence"> _edgePmrPrecedence = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "Edge/Connection resource outranks Graph profile"_test = [] {
        TrackingResource graphPool;
        TrackingResource edgePool;
        Graph            graph(ResourceProfile{.data = &graphPool, .tag = &graphPool});
        auto&            src  = graph.emplaceBlock<NullSource<float>>();
        auto&            sink = graph.emplaceBlock<NullSink<float>>();

        expect(graph.connect<"out", "in">(src, sink, {.minBufferSize = 4096UZ, .dataResource = &edgePool, .tagResource = &edgePool}).has_value());
        graph.connectPendingEdges();

        auto edges = graph.edges();
        expect(eq(edges.size(), 1UZ));
        expect(edges[0]._dataResource == &edgePool) << "Edge/Connection outranks Graph profile (data axis)";
        expect(edges[0]._tagResource == &edgePool) << "Edge/Connection outranks Graph profile (tag axis)";
    };

    "Graph profile used when no Edge/Connection resource"_test = [] {
        TrackingResource graphPool;
        Graph            graph(ResourceProfile{.data = &graphPool, .tag = &graphPool});
        auto&            src  = graph.emplaceBlock<NullSource<float>>();
        auto&            sink = graph.emplaceBlock<NullSink<float>>();

        expect(graph.connect<"out", "in">(src, sink, {.minBufferSize = 4096UZ}).has_value());
        graph.connectPendingEdges();

        auto edges = graph.edges();
        expect(eq(edges.size(), 1UZ));
        expect(edges[0]._dataResource == &graphPool) << "Graph profile used when Edge/Connection unset (data axis)";
        expect(edges[0]._tagResource == &graphPool) << "Graph profile used when Edge/Connection unset (tag axis)";
    };

    "Graph profile outranks non-host domain USM"_test = [] {
        TrackingResource graphPool;
        TrackingResource usmPool;
        ComputeRegistry::instance().register_provider("test-precedence-usm", &testProvider);
        Graph graph(ResourceProfile{.data = &graphPool, .tag = &graphPool});
        auto& src  = graph.emplaceBlock<NullSource<float>>();
        auto& sink = graph.emplaceBlock<NullSink<float>>();

        EdgeParameters params;
        params.minBufferSize = 4096UZ;
        params.domain        = ComputeDomain::gpu_shared("test-precedence-usm");
        params.domain.user   = &usmPool; // resolvable device USM — outranked by the explicit Graph profile
        expect(graph.connect<"out", "in">(src, sink, params).has_value());
        graph.connectPendingEdges();

        auto edges = graph.edges();
        expect(eq(edges.size(), 1UZ));
        expect(edges[0]._dataResource == &graphPool) << "Graph profile outranks non-host domain USM";
    };

    "host edge with nothing set falls back to global default"_test = [] {
        Graph graph;
        auto& src  = graph.emplaceBlock<NullSource<float>>();
        auto& sink = graph.emplaceBlock<NullSink<float>>();

        expect(graph.connect<"out", "in">(src, sink, {.minBufferSize = 4096UZ}).has_value());
        graph.connectPendingEdges();

        auto edges = graph.edges();
        expect(eq(edges.size(), 1UZ));
        expect(edges[0]._dataResource == std::pmr::get_default_resource()) << "global default when Edge/Graph/USM all unset";
    };
};

namespace group_blocks_test {

std::string registeredSimpleSchedulerType() {
    std::ignore = gr::globalSchedulerRegistry().insert<gr::scheduler::Simple<>>();
    return std::string(gr::meta::type_name<gr::scheduler::Simple<>>());
}

std::string exportedPortName(const gr::property_map& exportedPorts, std::string_view blockUniqueName, std::string_view internalPortName) {
    return exportedPorts //
        .value_or<gr::property_map>(gr::convert_string_domain(blockUniqueName), gr::property_map{})
        .value_or<gr::property_map>(gr::convert_string_domain(internalPortName), gr::property_map{})
        .value_or<std::string>("exportedName", std::string{});
}

// optional port names, if none are supplied then they will match all names and look for any edge
bool hasEdge(const gr::Graph& graph, const gr::BlockModel* sourceBlock, std::optional<std::string_view> sourcePortName, const gr::BlockModel* destinationBlock, std::optional<std::string_view> destinationPortName) {
    const auto portMatches = [](const gr::PortDefinition& definition, std::optional<std::string_view> name) {
        const auto* stringBased = std::get_if<gr::PortDefinition::StringBased>(&definition.definition);
        return !name.has_value() || (stringBased != nullptr && stringBased->name == *name);
    };
    return std::ranges::any_of(graph.edges(), [&](const gr::Edge& edge) { //
        return edge.sourceBlock().get() == sourceBlock && edge.destinationBlock().get() == destinationBlock && portMatches(edge.sourcePortDefinition(), sourcePortName) && portMatches(edge.destinationPortDefinition(), destinationPortName);
    });
}
constexpr std::optional<std::string_view> anyPortName; // matches any name

void testGroupSingleMiddleBlock(std::string_view subGraphType) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    Graph graph;
    auto& src  = graph.emplaceBlock<NullSource<float>>();
    auto& copy = graph.emplaceBlock<Copy<float>>();
    auto& sink = graph.emplaceBlock<NullSink<float>>();
    expect(graph.connect<"out", "in">(src, copy).has_value()) << fatal;
    expect(graph.connect<"out", "in">(copy, sink).has_value()) << fatal;

    const BlockModel* originalCopyModel = graph.blocks()[1UZ].get();
    const auto        srcModel          = graph::findBlock(graph, src).value();
    const auto        sinkModel         = graph::findBlock(graph, sink).value();

    const std::vector<std::string_view> uniqueNamesOfGroupedBlocks = {copy.unique_name.value()};
    auto                                groupedBlocks              = graph::findBlocks(graph, uniqueNamesOfGroupedBlocks).value();
    const auto                          grouped                    = graph.groupBlocks(groupedBlocks, subGraphType);
    expect(grouped.has_value()) << [&] { return grouped ? std::string{} : grouped.error().message; } << fatal;
    const std::shared_ptr<BlockModel>& subGraph = grouped.value();

    expect(graph::findBlock(graph, subGraph).has_value());
    expect(graph::findBlock(graph, src).has_value());
    expect(graph::findBlock(graph, sink).has_value());
    expect(eq(graph.blocks().size(), 3UZ)) << "graph should have replaced the Copy block with a subgraph";

    expect(eq(subGraph->blocks().size(), 1UZ)) << fatal;
    expect(eq(subGraph->edges().size(), 0UZ));
    expect(subGraph->blocks()[0UZ].get() == originalCopyModel) << "grouped block is the same block, just moved";

    expect(eq(subGraph->dynamicInputPortsSize(), 1UZ)) << "subgraph has exactly one exported input port";
    expect(eq(subGraph->dynamicOutputPortsSize(), 1UZ)) << "subgraph has exactly one exported output port";

    const std::string exportedIn  = exportedPortName(subGraph->exportedInputPorts(), copy.unique_name.value(), "in");
    const std::string exportedOut = exportedPortName(subGraph->exportedOutputPorts(), copy.unique_name.value(), "out");
    expect(!exportedIn.empty()) << "input port of the Copy block is exported";
    expect(!exportedOut.empty()) << "output port of the Copy block is exported";
    expect(subGraph->dynamicInputPort(PortDefinition(exportedIn)).has_value());
    expect(subGraph->dynamicOutputPort(PortDefinition(exportedOut)).has_value());

    expect(eq(graph.edges().size(), 2UZ));
    expect(hasEdge(graph, srcModel.get(), anyPortName, subGraph.get(), exportedIn)) << "src connects to the exported input port";
    expect(hasEdge(graph, subGraph.get(), exportedOut, sinkModel.get(), anyPortName)) << "exported output port connects to the sink";

    expect(graph.connectPendingEdges());
    expect(std::ranges::all_of(graph.edges(), [](const Edge& edge) { return edge.state() == Edge::EdgeState::Connected; }));
}

void testGroupNonAdjacentBlocks(std::string_view subGraphType) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    Graph graph;
    auto& src   = graph.emplaceBlock<NullSource<float>>();
    auto& copy2 = graph.emplaceBlock<Copy<float>>();
    auto& copy3 = graph.emplaceBlock<Copy<float>>();
    auto& copy4 = graph.emplaceBlock<Copy<float>>();
    auto& sink  = graph.emplaceBlock<NullSink<float>>();
    expect(graph.connect<"out", "in">(src, copy2).has_value()) << fatal;
    expect(graph.connect<"out", "in">(copy2, copy3).has_value()) << fatal;
    expect(graph.connect<"out", "in">(copy3, copy4).has_value()) << fatal;
    expect(graph.connect<"out", "in">(copy4, sink).has_value()) << fatal;

    // group two blocks that have a block in between them, which produces a somewhat confusing but valid result
    const BlockModel* originalCopy2Model = graph.blocks()[1UZ].get();
    const BlockModel* originalCopy4Model = graph.blocks()[3UZ].get();
    // make sure they were at the expected indices + we are testing the correct blocks
    expect(originalCopy2Model->uniqueName() == copy2.unique_name.value());
    expect(originalCopy4Model->uniqueName() == copy4.unique_name.value());
    const auto srcModel   = graph::findBlock(graph, src).value();
    const auto copy3Model = graph::findBlock(graph, copy3).value();
    const auto sinkModel  = graph::findBlock(graph, sink).value();

    const std::vector<std::string_view> uniqueNamesOfGroupedBlocks = {copy2.unique_name.value(), copy4.unique_name.value()};
    const auto                          groupedBlocks              = graph::findBlocks(graph, uniqueNamesOfGroupedBlocks).value();
    const auto                          grouped                    = graph.groupBlocks(groupedBlocks, subGraphType);
    expect(grouped.has_value()) << [&] { return grouped ? std::string{} : grouped.error().message; } << fatal;
    const std::shared_ptr<BlockModel>& subGraph = grouped.value();

    expect(eq(graph.blocks().size(), 4UZ)) << "parent graph contains source, middle block, sink and the subgraph";
    expect(eq(subGraph->blocks().size(), 2UZ)) << fatal;
    expect(std::ranges::any_of(subGraph->blocks(), [&](const auto& block) { return block.get() == originalCopy2Model; })) << "grouped block 2 is moved, not copied";
    expect(std::ranges::any_of(subGraph->blocks(), [&](const auto& block) { return block.get() == originalCopy4Model; })) << "grouped block 4 is moved, not copied";
    expect(eq(subGraph->edges().size(), 0UZ)) << "non-adjacent blocks are/were not connected";

    expect(eq(subGraph->dynamicInputPortsSize(), 2UZ)) << "subgraph has two exported input ports";
    expect(eq(subGraph->dynamicOutputPortsSize(), 2UZ)) << "subgraph has two exported output ports";

    const std::string exportedIn2  = exportedPortName(subGraph->exportedInputPorts(), copy2.unique_name.value(), "in");
    const std::string exportedOut2 = exportedPortName(subGraph->exportedOutputPorts(), copy2.unique_name.value(), "out");
    const std::string exportedIn4  = exportedPortName(subGraph->exportedInputPorts(), copy4.unique_name.value(), "in");
    const std::string exportedOut4 = exportedPortName(subGraph->exportedOutputPorts(), copy4.unique_name.value(), "out");
    expect(!exportedIn2.empty() && !exportedOut2.empty() && !exportedIn4.empty() && !exportedOut4.empty()) << "all boundary ports are exported" << fatal;
    expect(exportedIn2 != exportedIn4) << "input port names should be unique";
    expect(exportedOut2 != exportedOut4) << "output port names should be unique";
    expect(subGraph->dynamicInputPort(PortDefinition(exportedIn4)).has_value());
    expect(subGraph->dynamicOutputPort(PortDefinition(exportedOut4)).has_value());
    expect(subGraph->dynamicInputPort(PortDefinition(exportedIn2)).has_value());
    expect(subGraph->dynamicOutputPort(PortDefinition(exportedOut2)).has_value());

    expect(eq(graph.edges().size(), 4UZ));
    expect(hasEdge(graph, srcModel.get(), anyPortName, subGraph.get(), exportedIn2)) << "source connects to the exported input port of block 2";
    expect(hasEdge(graph, subGraph.get(), exportedOut2, copy3Model.get(), anyPortName)) << "exported output port of block 2 connects to block 3";
    expect(hasEdge(graph, copy3Model.get(), anyPortName, subGraph.get(), exportedIn4)) << "block 3 connects to the exported input port of block 4";
    expect(hasEdge(graph, subGraph.get(), exportedOut4, sinkModel.get(), anyPortName)) << "exported output port of block 4 connects to the sink";

    expect(graph.connectPendingEdges());
    expect(std::ranges::all_of(graph.edges(), [](const Edge& edge) { return edge.state() == Edge::EdgeState::Connected; }));
}

void testGroupAdjacentBlocks(std::string_view subGraphType) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    Graph graph;
    auto& src   = graph.emplaceBlock<NullSource<float>>();
    auto& copy2 = graph.emplaceBlock<Copy<float>>();
    auto& copy3 = graph.emplaceBlock<Copy<float>>();
    auto& sink  = graph.emplaceBlock<NullSink<float>>();
    expect(graph.connect<"out", "in">(src, copy2).has_value()) << fatal;
    expect(graph.connect<"out", "in">(copy2, copy3).has_value()) << fatal;
    expect(graph.connect<"out", "in">(copy3, sink).has_value()) << fatal;

    const BlockModel* originalCopy2Model = graph.blocks()[1UZ].get();
    const BlockModel* originalCopy3Model = graph.blocks()[2UZ].get();

    const std::vector<std::string_view> uniqueNamesOfGroupedBlocks = {copy2.unique_name.value(), copy3.unique_name.value()};
    const auto                          groupedBlocks              = graph::findBlocks(graph, uniqueNamesOfGroupedBlocks).value();
    const auto                          grouped                    = graph.groupBlocks(groupedBlocks, subGraphType);
    expect(grouped.has_value()) << [&] { return grouped ? std::string{} : grouped.error().message; } << fatal;
    const std::shared_ptr<BlockModel>& subGraph = grouped.value();

    expect(eq(graph.blocks().size(), 3UZ));
    expect(eq(graph.edges().size(), 2UZ));
    expect(eq(subGraph->blocks().size(), 2UZ));
    expect(eq(subGraph->edges().size(), 1UZ)) << "the edge between the grouped blocks moves into the subgraph" << fatal;
    expect(subGraph->edges()[0UZ].sourceBlock().get() == originalCopy2Model);
    expect(subGraph->edges()[0UZ].destinationBlock().get() == originalCopy3Model);

    expect(eq(subGraph->dynamicInputPortsSize(), 1UZ)) << "only the boundary-crossing input is exported";
    expect(eq(subGraph->dynamicOutputPortsSize(), 1UZ)) << "only the boundary-crossing output is exported";

    expect(graph.connectPendingEdges());
    expect(std::ranges::all_of(graph.edges(), [](const Edge& edge) { return edge.state() == Edge::EdgeState::Connected; }));
}

const boost::ut::suite<"Graph::groupBlocks"> _groupBlocks = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "group single middle block into unmanaged gr::Graph"_test = [] { testGroupSingleMiddleBlock("gr::Graph"); };
    "group single middle block into managed scheduler"_test   = [] { testGroupSingleMiddleBlock(registeredSimpleSchedulerType()); };

    "group two non-adjacent blocks into unmanaged gr::Graph"_test = [] { testGroupNonAdjacentBlocks("gr::Graph"); };
    "group two non-adjacent blocks into managed scheduler"_test   = [] { testGroupNonAdjacentBlocks(registeredSimpleSchedulerType()); };

    "group two adjacent blocks into unmanaged gr::Graph"_test = [] { testGroupAdjacentBlocks("gr::Graph"); };
    "group two adjacent blocks into managed scheduler"_test   = [] { testGroupAdjacentBlocks(registeredSimpleSchedulerType()); };

    "graph::findBlocks() fails for unknown block names"_test = [] {
        Graph graph;
        auto& src   = graph.emplaceBlock<NullSource<float>>();
        std::ignore = src;

        const std::vector<std::string_view> uniqueNamesOfGroupedBlocks = {src.unique_name.value(), "foo"};
        const auto                          groupedBlocks              = graph::findBlocks(graph, uniqueNamesOfGroupedBlocks);
        expect(!groupedBlocks.has_value());
    };

    "grouping fails for unknown subgraph type"_test = [] {
        Graph graph;
        auto& src = graph.emplaceBlock<NullSource<float>>();

        const std::vector<std::string_view> uniqueNamesOfGroupedBlocks = {src.unique_name.value()};
        const auto                          groupedBlocks              = graph::findBlocks(graph, uniqueNamesOfGroupedBlocks).value();
        expect(!graph.groupBlocks(groupedBlocks, "does::not::Exist").has_value());
        expect(eq(graph.blocks().size(), 1UZ)) << "failed grouping leaves the graph unchanged";
    };
};

} // namespace group_blocks_test

int main() { /* not needed for UT */ }
