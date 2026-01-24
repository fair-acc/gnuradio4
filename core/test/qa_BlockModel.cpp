#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockModel.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

#include <format>
#include <unordered_map>

#include "message_utils.hpp"

using namespace boost::ut;
using namespace std::string_literals;

namespace gr::test {

struct SimpleBlock : Block<SimpleBlock, Resampling<2, 1, false>, Stride<4>> {
    using Description = Doc<R""(@brief synchronous single in/out, const resampling ratio, runtime stride)"">;

    PortIn<int>  in{};
    PortOut<int> out{};

    GR_MAKE_REFLECTABLE(SimpleBlock, in, out);

    work::Status processBulk(InputSpanLike auto& inputSpan, OutputSpanLike auto& outputSpan) const {
        if (const auto n = std::min(inputSpan.size(), outputSpan.size()); n > 0UZ) {
            std::copy_n(inputSpan.begin(), n, outputSpan.begin());
            outputSpan.publish(n);
        }
        return work::Status::DONE;
    }
};
static_assert(BlockLike<SimpleBlock>);

struct VectorPortBlock : Block<VectorPortBlock> {
    using Description = Doc<R""(@brief Dynamic port collections to test NamedPortCollection branches)"">;

    Size_t                             n_ports = 3U;
    std::vector<PortIn<float, Async>>  vin{n_ports};
    std::vector<PortOut<float, Async>> vout{n_ports};

    GR_MAKE_REFLECTABLE(VectorPortBlock, n_ports, vin, vout);

    template<InputSpanLike TIn, OutputSpanLike TOut>
    work::Status processBulk(std::span<TIn>& inputs, std::span<TOut>& outputs) const {
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto n = std::min(inputs[i].size(), outputs[i].size());
            std::copy_n(inputs[i].begin(), n, outputs[i].begin());
            std::ignore = inputs[i].consume(n);
            outputs[i].publish(n);
        }
        return work::Status::DONE;
    }
};
static_assert(BlockLike<VectorPortBlock>);

struct DrawableBlock : Block<DrawableBlock, Drawable<UICategory::Toolbar, "console">> {
    using Description = Doc<R""(@brief Drawable block to exercise draw()/uiCategory()/blockCategory())"">;

    PortOut<float> out{};

    GR_MAKE_REFLECTABLE(DrawableBlock, out);

    work::Status draw(const property_map& config = {}, std::source_location location = std::source_location::current()) const noexcept {
        std::println("DrawableBlock draw({}) called at {}", config, location);
        return work::Status::OK;
    }
    float processOne() const { return 1.f; }
};
static_assert(BlockLike<DrawableBlock>);

template<typename T, std::size_t nInSync, std::size_t nInAsync, std::size_t nMsgIn, std::size_t nOutSync, std::size_t nOutAsync, std::size_t nMsgOut>
requires(std::is_arithmetic_v<T>)
struct GenericBlock : gr::Block<GenericBlock<T, nInSync, nInAsync, nMsgIn, nOutSync, nOutAsync, nMsgOut>> {
    std::array<gr::PortIn<T>, nInSync>               in1{};
    std::array<gr::PortIn<T, gr::Async>, nInAsync>   in2{};
    std::array<gr::PortOut<T>, nOutSync>             out1{};
    std::array<gr::PortOut<T, gr::Async>, nOutAsync> out2{};

    std::array<gr::MsgPortIn, nMsgIn>   msgIns{};
    std::array<gr::MsgPortOut, nMsgOut> msgOuts{};

    GR_MAKE_REFLECTABLE(GenericBlock, in1, in2, out1, out2, msgIns, msgOuts);

    gr::work::Status processBulk(auto /*ins1*/, auto /*ins2*/, auto /*outs1*/, auto /*outs2*/) { return gr::work::Status::OK; }
};

} // namespace gr::test

const suite<"BlockModel API"> _1 = [] { // NOSONAR (N.B. lambda size)
    using namespace gr;
    using namespace gr::testing;
    using gr::test::SimpleBlock;
    using gr::test::VectorPortBlock;
    using gr::test::DrawableBlock;

    "dynamic ports: init, name/index/PortDefinition lookups, indices & sizes, protected helper"_test = [] { // NOSONAR (N.B. lambda size)
        std::shared_ptr<Sequence>     _progress;
        BlockWrapper<VectorPortBlock> dynBlock;

        auto rawBlockRef = static_cast<VectorPortBlock*>(dynBlock.raw());
        rawBlockRef->init(_progress);

        std::vector<BlockWrapper<VectorPortBlock>::DynamicPortOrCollection>& inputCollections  = dynBlock.dynamicInputPorts();  // also implicitly initialises dynamic input ports
        std::vector<BlockWrapper<VectorPortBlock>::DynamicPortOrCollection>& outputCollections = dynBlock.dynamicOutputPorts(); // also implicitly initialises dynamic input ports

        expect(eq(inputCollections.size(), 1UZ));
        expect(eq(outputCollections.size(), 1UZ));

        expect(eq(dynBlock.dynamicInputPortsSize(), 1UZ));
        expect(eq(dynBlock.dynamicOutputPortsSize(), 1UZ));
        expect(eq(dynBlock.dynamicInputPortsSize(0), rawBlockRef->n_ports));
        expect(eq(dynBlock.dynamicOutputPortsSize(0), rawBlockRef->n_ports));

        const DynamicPort& inputPort1ByString = dynBlock.dynamicInputPort("vin#1");
        expect(eq(inputPort1ByString.name, "vin"s)); // user would probably expect 'vin#1'

        const DynamicPort& outputPort2ByIndex = dynBlock.dynamicOutputPort(0, 2);
        expect(eq(outputPort2ByIndex.name, "vout"s)); // user would probably expect 'vout#2'

        PortDefinition outputDefIdx{0UZ, 0UZ};
        PortDefinition outputDefStr{"vout#0"};
        expect(eq(dynBlock.dynamicOutputPort(outputDefIdx).portName(), "vout"s)); // user would probably expect 'vout#0'
        expect(eq(dynBlock.dynamicOutputPort(outputDefStr).portName(), "vout"s)); // user would probably expect 'vout#0'

        const std::size_t outIndex = dynBlock.dynamicOutputPortIndex("vout");
        expect(eq(outIndex, 0UZ));

        const std::size_t inIndex = dynBlock.dynamicInputPortIndex("vin");
        expect(eq(inIndex, 0UZ));

        expect(eq(BlockModel::portName(outputCollections[0]), "vout"s)) << "portName utility";

        // error paths
        expect(throws([&dynBlock] { std::ignore = dynBlock.dynamicInputPort("does_not_exist"); })) << "invalid named port";
        expect(throws([&dynBlock] { std::ignore = dynBlock.dynamicInputPort(1UZ, 0UZ); })) << "invalid input port index";
        expect(throws([&dynBlock] { std::ignore = dynBlock.dynamicOutputPort(1UZ, 0UZ); })) << "invalid output port index";
        expect(throws([&dynBlock] { std::ignore = dynBlock.dynamicInputPort(0UZ, 99UZ); })) << "invalid input port sub-index";
        expect(throws([&dynBlock] { std::ignore = dynBlock.dynamicOutputPort(0UZ, 99UZ); })) << "invalid output port sub-index";
        expect(throws([&dynBlock] { std::ignore = dynBlock.dynamicInputPort(42UZ); })) << "invalid input port sub-index";
        expect(throws([&dynBlock] { std::ignore = dynBlock.dynamicOutputPort(42UZ); })) << "invalid output port sub-index";
    };

    "ratios/stride, masks, async flags"_test = [] {
        BlockWrapper<SimpleBlock> dynBlock;
        expect(!dynBlock.dynamicInputPorts().empty());  // also implicitly initialises dynamic input ports
        expect(!dynBlock.dynamicOutputPorts().empty()); // also implicitly initialises dynamic input ports

        expect(eq(dynBlock.resamplingRatio().num(), 2));
        expect(eq(dynBlock.resamplingRatio().den(), 1));
        expect(eq(dynBlock.stride(), 4UZ));

        auto inputMasks  = dynBlock.blockInputTypes();
        auto outputMasks = dynBlock.blockOutputTypes();
        expect(eq(inputMasks.size(), 1UZ));
        expect(eq(outputMasks.size(), 1UZ));

        expect(!dynBlock.hasAsyncInputPorts());
        expect(!dynBlock.hasAsyncOutputPorts());
    };

    "settings/metaInformation/uiConstraints getters & setters"_test = [] {
        BlockWrapper<SimpleBlock> dynBlock;

        expect(!dynBlock.metaInformation().contains("answer"));
        dynBlock.metaInformation()["answer"] = 42;
        expect(eq(get_value_or_fail<int>(dynBlock.metaInformation().at("answer")), 42));

        expect(dynBlock.uiConstraints().empty());
        dynBlock.uiConstraints()["x-position"] = 3.0f;
        dynBlock.uiConstraints()["y-position"] = 4.0f;
        expect(eq(get_value_or_fail<float>(dynBlock.uiConstraints().at("x-position")), 3.0f));
        expect(eq(get_value_or_fail<float>(dynBlock.uiConstraints().at("y-position")), 4.0f));

        const property_map setReturn = dynBlock.settings().set({{"output_chunk_size", Size_t(5)}, {"input_chunk_size", Size_t(3)}});
        expect(setReturn.empty()) << std::format("set return not empty: {}", setReturn);
        expect(dynBlock.settings().activateContext() != std::nullopt);
        expect(eq(dynBlock.resamplingRatio().num(), 2));
        expect(eq(dynBlock.resamplingRatio().den(), 1));
    };

    "state machine, name/type/uniqueName, categories, uiCategory, draw(), processScheduledMessages()"_test = [] { // NOSONAR (N.B. lambda size)
        BlockWrapper<DrawableBlock> dynBlock;

        expect(dynBlock.blockCategory() == block::Category::NormalBlock);
        expect(dynBlock.uiCategory() == UICategory::Toolbar);
        expect(!dynBlock.isBlocking());

        auto initialName = dynBlock.name();
        expect(!initialName.empty());

        dynBlock.setName("new_runtime_name");
        expect(eq(dynBlock.name(), "new_runtime_name"sv));
        expect(!dynBlock.uniqueName().empty());
        expect(!dynBlock.typeName().empty());

        using enum gr::lifecycle::State;
        expect(dynBlock.state() == IDLE);
        expect(dynBlock.changeStateTo(INITIALISED).has_value());
        expect(dynBlock.state() == INITIALISED);
        expect(dynBlock.changeStateTo(RUNNING).has_value());
        expect(dynBlock.state() == RUNNING);

        expect(dynBlock.draw() == work::Status::OK) << "draw() reachable";

        // processScheduledMessages() no-op but callable
        dynBlock.processScheduledMessages();

        expect(dynBlock.changeStateTo(REQUESTED_STOP).has_value());
        expect(dynBlock.changeStateTo(STOPPED).has_value());
        expect(dynBlock.state() == STOPPED);
    };

    "work(), availableSamples()/requirements(), primeInputPort() with downstream verify"_test = [] { // NOSONAR (N.B. lambda size)
        Graph            processingGraph;
        constexpr Size_t kNumSamples = 128;

        auto& sourceBlock    = processingGraph.emplaceBlock<TagSource<int>>({{"n_samples_max", kNumSamples}, {"mark_tag", false}});
        auto& middleBlock    = processingGraph.emplaceBlock<SimpleBlock>();
        auto& sinkBlock      = processingGraph.emplaceBlock<TagSink<int, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});
        auto  middleBlockPtr = graph::findBlock(processingGraph, middleBlock).value();
        auto  sinkBlockPtr   = graph::findBlock(processingGraph, sinkBlock).value();

        expect(processingGraph.connect<"out">(sourceBlock).to<"in">(middleBlock) == ConnectionResult::SUCCESS);
        expect(processingGraph.connect<"out">(middleBlock).to<"in">(sinkBlock) == ConnectionResult::SUCCESS);
        expect(!processingGraph.blocks().empty());

        // Prime data into middle block input before any scheduler run
        constexpr static std::size_t kPrimeCount           = 16UZ;
        "test primeInputPort() with unconnected port"_test = [&middleBlockPtr] {
            std::expected<std::size_t, Error> primeResult = middleBlockPtr->primeInputPort(0, kPrimeCount);
            expect(!primeResult.has_value()) << [&primeResult] { return std::format("primeInputPort() did not fail returned: {}", primeResult.value()); };
        };

        "test primeInputPort() with unconnected port"_test = [&processingGraph, &middleBlockPtr] {
            processingGraph.connectPendingEdges(); // needed
            std::expected<std::size_t, Error> primeResult = middleBlockPtr->primeInputPort(0, kPrimeCount);
            expect(primeResult.has_value()) << [&primeResult] { return std::format("primeInputPort() failed: {}", primeResult.error()); };
            expect(eq(primeResult.value(), kPrimeCount));

            auto middleAvailAfterPrime = middleBlockPtr->availableInputSamples(true);
            expect(eq(middleAvailAfterPrime[0UZ], kPrimeCount));
        };

        auto sinkAvailBeforeWork = sinkBlockPtr->availableInputSamples(true);
        expect(eq(sinkAvailBeforeWork[0UZ], 0UZ)) << "downstream block (sink) should see zero samples (no processing done)";
    };

    "input/output meta infos"_test = [] {
        BlockWrapper<SimpleBlock> simpleWrapper;
        // meta infos are cached by port caches; reset=true forces reading
        auto inputInfos  = simpleWrapper.inputMetaInfos(true);
        auto outputInfos = simpleWrapper.outputMetaInfos(true);
        expect(eq(inputInfos.size(), 1UZ));
        expect(eq(outputInfos.size(), 1UZ));
    };

    "blocks()/edges() traversal on nested graphs"_test = [] {
        auto wrappedRootGraph = std::make_shared<BlockWrapper<Graph>>();
        auto rootGraph        = static_cast<Graph*>(wrappedRootGraph->raw());

        auto& sourceBlock = rootGraph->emplaceBlock<NullSource<float>>();
        auto& nestedGraph = rootGraph->emplaceBlock<Graph>();
        auto& sinkBlock   = nestedGraph.emplaceBlock<NullSink<float>>();

        "find blocks in flattened graph"_test = [rootGraph, &sourceBlock, &sinkBlock] {
            Graph flatGraph = graph::flatten(*rootGraph);
            auto  src       = graph::findBlock(flatGraph, sourceBlock);
            auto  dst       = graph::findBlock(flatGraph, sinkBlock);
            expect(src.has_value()) << [&src] { return std::format("sourceBlock not found in flattened graph: {}", src.error()); };
            expect(dst.has_value()) << [&dst] { return std::format("sinkBlock not found in flattened graph: {}", dst.error()); };
        };

        // FIXME: discuss how to connect from/to blocks in different (sub-)graphs
        skip / "connect source (in root graph) with sink (in nested graph)"_test = [&rootGraph, &sourceBlock, &sinkBlock, &wrappedRootGraph] {
            expect(eq(rootGraph->connect<"out">(sourceBlock).to<"in">(sinkBlock), ConnectionResult::SUCCESS));
            rootGraph->connectPendingEdges();
            expect(!wrappedRootGraph->blocks().empty());
            expect(eq(wrappedRootGraph->edges().size(), 1UZ));
        };

        "two disjoint chains + isolate"_test = [] {
            gr::Graph graph;

            // sub-graph#1
            auto& s1 = graph.emplaceBlock<gr::testing::NullSource<float>>();
            auto& m1 = graph.emplaceBlock<gr::testing::NullSink<float>>();
            expect(graph.connect<"out">(s1).to<"in">(m1) == gr::ConnectionResult::SUCCESS);
            // sub-graph#2
            auto& s2 = graph.emplaceBlock<gr::testing::NullSource<float>>();
            auto& m2 = graph.emplaceBlock<gr::testing::NullSink<float>>();
            expect(graph.connect<"out">(s2).to<"in">(m2) == gr::ConnectionResult::SUCCESS);
            // sub-graph#3 -- only one block, no connections
            std::ignore = graph.emplaceBlock<gr::testing::NullSource<float>>();

            std::vector<gr::Graph> subgraphs = gr::graph::weaklyConnectedComponents(graph);

            using std::views::transform;
            using std::ranges::to;
            const auto sizes = subgraphs | transform([](const auto& sg) { return sg.blocks().size(); }) | to<std::vector<std::size_t>>();

            expect(eq(sizes.size(), 3UZ));
            expect(std::ranges::is_sorted(sizes, std::greater{}));
            expect(eq(sizes, std::vector{2UZ, 2UZ, 1UZ}));

            const auto esz = subgraphs | transform([](const auto& sg) { return sg.edges().size(); }) | to<std::vector<std::size_t>>();
            expect(eq(esz, std::vector{1UZ, 1UZ, 0UZ}));
        };
    };

    "raw() pointer exposes underlying block object"_test = [] {
        BlockWrapper<SimpleBlock> simpleWrapper;
        auto*                     underlying = static_cast<SimpleBlock*>(simpleWrapper.raw());
        expect(underlying != nullptr);
        expect(eq(&underlying->in, &static_cast<SimpleBlock*>(simpleWrapper.raw())->in));
    };

    "PortDefinition hashing & equality in unordered_map"_test = [] {
        PortDefinition defIndexA{1UZ, 2UZ};
        PortDefinition defIndexB{1UZ, 2UZ};
        PortDefinition defStringA{"foo"};
        PortDefinition defStringB{"foo"};

        expect(defIndexA == defIndexB);
        expect(defStringA == defStringB);

        std::unordered_map<PortDefinition, int> portMap;
        portMap[defIndexA]  = 10;
        portMap[defStringA] = 20;

        expect(eq(portMap[defIndexB], 10));
        expect(eq(portMap[defStringB], 20));
    };

    "Edge hash/equality/containers"_test = [] {
        gr::Graph  graph;
        const auto n    = graph.blocks().size();
        std::ignore     = graph.emplaceBlock<gr::testing::NullSource<float>>();
        std::ignore     = graph.emplaceBlock<gr::testing::NullSink<float>>();
        const auto& src = graph.blocks()[n + 0];
        const auto& dst = graph.blocks()[n + 1];

        gr::Edge e1{src, gr::PortDefinition{"out"}, dst, gr::PortDefinition{"in"}, 0UZ, 0, "a"};
        gr::Edge e2{src, gr::PortDefinition{"out"}, dst, gr::PortDefinition{"in"}, 64UZ, 7, "b"}; // equal key
        gr::Edge e3{src, gr::PortDefinition{"out"}, dst, gr::PortDefinition{"in2"}, 0UZ, 0, "c"}; // different key

        expect(e1 == e2);
        expect(e1 != e3);

        expect(eq(std::hash<gr::Edge>{}(e1), std::hash<gr::Edge>{}(e2)));
        expect(not eq(std::hash<gr::Edge>{}(e1), std::numeric_limits<std::size_t>::max()));
        expect(not eq(std::hash<gr::Edge>{}(e1), 0UZ));
        expect(not eq(std::hash<gr::Edge>{}(e3), 0UZ));
        expect(not eq(std::hash<gr::Edge>{}(e3), std::numeric_limits<std::size_t>::max()));

        std::unordered_map<gr::Edge, int> m{{e1, 42}};
        expect(eq(m[e2], 42));

        std::unordered_set<gr::Edge> s{e1, e2, e3};
        expect(eq(s.size(), 2UZ));
    };
};

const suite<"PortIndexTests"> _2 = [] {
    using namespace gr::test;

    "port index resolution - single ports"_test = []<typename T>(T) {
        constexpr static std::size_t nPortsBefore = T();

        gr::Graph graph;
        auto&     block      = graph.emplaceBlock<GenericBlock<float, nPortsBefore, 1, 0, 1, 0, 0>>({{"name", "single"}});
        auto      blockModel = gr::graph::findBlock(graph, block.unique_name).value();

        "input ports"_test = [&blockModel] {
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{1UZ}), nPortsBefore + 0UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{"in2#0"}), nPortsBefore + 0UZ));
        };

        "output ports"_test = [&blockModel] {
            expect(eq(gr::absolutePortIndex<gr::PortDirection::OUTPUT>(blockModel, gr::PortDefinition{0UZ}), 0UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::OUTPUT>(blockModel, gr::PortDefinition{"out1#0"}), 0UZ));
        };
    } | std::tuple<std::integral_constant<std::size_t, 0UZ>, std::integral_constant<std::size_t, 1UZ>, std::integral_constant<std::size_t, 2UZ>>{};

    "port index resolution - multiple sync ports"_test = []<typename T>(T) {
        constexpr static std::size_t nPortsBefore = T();

        gr::Graph graph;
        auto&     block      = graph.emplaceBlock<GenericBlock<float, 3, 0, 0, nPortsBefore, 2, 0>>({{"name", "multi_sync"}});
        auto      blockModel = gr::graph::findBlock(graph, block.unique_name).value();

        "input ports"_test = [&blockModel] {
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{0UZ, 0UZ}), 0UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{0UZ, 1UZ}), 1UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{0UZ, 2UZ}), 2UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{"in1#0"}), 0UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{"in1#1"}), 1UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{"in1#2"}), 2UZ));
        };

        "output ports"_test = [&blockModel] {
            expect(eq(gr::absolutePortIndex<gr::PortDirection::OUTPUT>(blockModel, gr::PortDefinition{1UZ, 0UZ}), nPortsBefore + 0UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::OUTPUT>(blockModel, gr::PortDefinition{1UZ, 1UZ}), nPortsBefore + 1UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::OUTPUT>(blockModel, gr::PortDefinition{"out2#0"}), nPortsBefore + 0UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::OUTPUT>(blockModel, gr::PortDefinition{"out2#1"}), nPortsBefore + 1UZ));
        };
    } | std::tuple<std::integral_constant<std::size_t, 0UZ>, std::integral_constant<std::size_t, 1UZ>, std::integral_constant<std::size_t, 2UZ>>{};
    ;

    "port index resolution - mixed sync and async"_test = [] {
        gr::Graph graph;
        auto&     block      = graph.emplaceBlock<GenericBlock<float, 2, 3, 0, 1, 2, 0>>({{"name", "mixed"}});
        auto      blockModel = gr::graph::findBlock(graph, block.unique_name).value();

        "input ports: 2 sync (in1), then 3 async (in2)"_test = [&blockModel] {
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{0UZ, 0UZ}), 0UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{0UZ, 1UZ}), 1UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{1UZ, 0UZ}), 2UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{1UZ, 1UZ}), 3UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{1UZ, 2UZ}), 4UZ));
        };

        "string-based"_test = [&blockModel] {
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{"in1#0"}), 0UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{"in1#1"}), 1UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{"in2#0"}), 2UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{"in2#1"}), 3UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{"in2#2"}), 4UZ));
        };

        "output ports: 1 sync (out1), then 2 async (out2)"_test = [&blockModel] {
            expect(eq(gr::absolutePortIndex<gr::PortDirection::OUTPUT>(blockModel, gr::PortDefinition{0UZ, 0UZ}), 0UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::OUTPUT>(blockModel, gr::PortDefinition{1UZ, 0UZ}), 1UZ));
            expect(eq(gr::absolutePortIndex<gr::PortDirection::OUTPUT>(blockModel, gr::PortDefinition{1UZ, 1UZ}), 2UZ));
        };

        expect(eq(gr::absolutePortIndex<gr::PortDirection::OUTPUT>(blockModel, gr::PortDefinition{"out1#0"}), 0UZ));
        expect(eq(gr::absolutePortIndex<gr::PortDirection::OUTPUT>(blockModel, gr::PortDefinition{"out2#0"}), 1UZ));
        expect(eq(gr::absolutePortIndex<gr::PortDirection::OUTPUT>(blockModel, gr::PortDefinition{"out2#1"}), 2UZ));
    };

    "flattening invariant"_test = [] {
        gr::Graph g;
        auto&     b  = g.emplaceBlock<GenericBlock<float, 2, 3, 0, 1, 2, 0>>({{"name", "inv"}}); // in: 2 then 3
        auto      bm = gr::graph::findBlock(g, b.unique_name).value();

        // expected ordinals: in1#0=0, in1#1=1, in2#0=2, in2#1=3, in2#2=4
        for (std::size_t k = 0; k < 2; ++k) {
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(bm, {0UZ, k}), k));
        }
        for (std::size_t k = 0; k < 3; ++k) {
            expect(eq(gr::absolutePortIndex<gr::PortDirection::INPUT>(bm, {1UZ, k}), 2UZ + k));
        }
    };

    "port index resolution - edge cases"_test = [] {
        "invalid PortDefinitions"_test = [] {
            gr::Graph graph;
            auto&     block      = graph.emplaceBlock<GenericBlock<float, 1, 1, 1, 1, 1, 1>>({{"name", "edge_cases"}});
            auto      blockModel = gr::graph::findBlock(graph, block.unique_name).value();

            expect(throws<gr::exception>([&blockModel] { std::ignore = gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{"nonexistent"}); }));
            expect(throws<gr::exception>([&blockModel] { std::ignore = gr::absolutePortIndex<gr::PortDirection::OUTPUT>(blockModel, gr::PortDefinition{"invalid_port"}); }));
            expect(throws<gr::exception>([&blockModel] { std::ignore = gr::absolutePortIndex<gr::PortDirection::INPUT>(blockModel, gr::PortDefinition{99UZ}); }));
        };

        "scalar hash OOB"_test = [] {
            gr::Graph g;
            auto&     b  = g.emplaceBlock<GenericBlock<float, 1, 0, 0, 1, 0, 0>>({{"name", "single"}});
            auto      bm = gr::graph::findBlock(g, b.unique_name).value();
            expect(throws<gr::exception>([&] { std::ignore = gr::absolutePortIndex<gr::PortDirection::INPUT>(bm, {"in1#1"}); }));
            expect(throws<gr::exception>([&] { std::ignore = gr::absolutePortIndex<gr::PortDirection::OUTPUT>(bm, {"out1#2"}); }));
        };

        "collection missing subindex"_test = [] {
            gr::Graph g;
            auto&     b  = g.emplaceBlock<GenericBlock<float, 0, 3, 0, 0, 2, 0>>({{"name", "coll"}}); // in2 size 3, out2 size 2
            auto      bm = gr::graph::findBlock(g, b.unique_name).value();
            expect(throws<gr::exception>([&] { std::ignore = gr::absolutePortIndex<gr::PortDirection::INPUT>(bm, {"in2"}); }));
            expect(throws<gr::exception>([&] { std::ignore = gr::absolutePortIndex<gr::PortDirection::OUTPUT>(bm, {"out2"}); }));
        };

        "indexbased subindex OOB"_test = [] {
            gr::Graph g;
            auto&     b  = g.emplaceBlock<GenericBlock<float, 0, 2, 0, 0, 1, 0>>({{"name", "oob"}}); // in2 size 2, out2 size 1
            auto      bm = gr::graph::findBlock(g, b.unique_name).value();
            expect(throws<gr::exception>([&] { std::ignore = gr::absolutePortIndex<gr::PortDirection::INPUT>(bm, {1UZ, 2UZ}); }));  // in2#2 OOB
            expect(throws<gr::exception>([&] { std::ignore = gr::absolutePortIndex<gr::PortDirection::OUTPUT>(bm, {1UZ, 1UZ}); })); // out2#1 OOB
        };

        "indexbased toplevel OOB"_test = [] {
            gr::Graph g;
            auto&     b  = g.emplaceBlock<GenericBlock<float, 1, 0, 0, 1, 0, 0>>({{"name", "oob2"}});
            auto      bm = gr::graph::findBlock(g, b.unique_name).value();
            expect(throws<gr::exception>([&] { std::ignore = gr::absolutePortIndex<gr::PortDirection::OUTPUT>(bm, {42UZ}); }));
        };

        "invalid hash tail rejects"_test = [] {
            gr::Graph g;
            auto&     b  = g.emplaceBlock<GenericBlock<float, 0, 3, 0, 0, 0, 0>>({{"name", "coll"}});
            auto      bm = gr::graph::findBlock(g, b.unique_name).value();
            // negative / non-numeric / trailing space should throw
            expect(throws<gr::exception>([&] { std::ignore = gr::absolutePortIndex<gr::PortDirection::INPUT>(bm, {"in2#-1"}); }));
            expect(throws<gr::exception>([&] { std::ignore = gr::absolutePortIndex<gr::PortDirection::INPUT>(bm, {"in2#abc"}); }));
            expect(throws<gr::exception>([&] { std::ignore = gr::absolutePortIndex<gr::PortDirection::INPUT>(bm, {"in2#1 "}); }));
        };
    };
};

int main() { /* not needed */ }
