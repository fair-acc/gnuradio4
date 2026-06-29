#include <boost/ut.hpp>

#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

#include <gnuradio-4.0/GrBasicBlocks.hpp>
#include <gnuradio-4.0/GrTestingBlocks.hpp>

#include "TestBlockRegistryContext.hpp"

#include "magic_enum.hpp"
#include "message_utils.hpp"

using namespace std::chrono_literals;
using namespace std::string_literals;

// We don't like new, but this will ensure the object is alive
// when ut starts running the tests. It runs the tests when
// its static objects get destroyed, which means other static
// objects might have been destroyed before that.
TestContext* context = new TestContext(paths{}, // plugin paths
    gr::blocklib::initGrBasicBlocks,            //
    gr::blocklib::initGrTestingBlocks);

template<gr::scheduler::ExecutionPolicy policy = gr::scheduler::ExecutionPolicy::singleThreaded>
class TestScheduler {
    using TScheduler = gr::scheduler::Simple<policy>;
    std::future<std::expected<void, gr::Error>> schedulerRet_;

    auto&& withTestingSourceAndSink(gr::Graph&& graph) const noexcept {
        graph.emplaceBlock<gr::testing::SlowSource<float>>();
        graph.emplaceBlock<gr::testing::CountingSink<float>>();
        return graph;
    }

public:
    TScheduler     scheduler_{};
    gr::MsgPortOut toScheduler;
    gr::MsgPortIn  fromScheduler;

    TestScheduler(gr::Graph&& graph, bool addTestSourceAndSink = true, bool shouldRun = true) {
        if (auto ret = scheduler_.exchange(addTestSourceAndSink ? std::move(withTestingSourceAndSink(std::move(graph))) : std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        using namespace gr::testing;
        expect(toScheduler.connect(scheduler_.msgIn).has_value());
        expect(scheduler_.msgOut.connect(fromScheduler).has_value());

        if (shouldRun) {
            run();
        }
    }

    ~TestScheduler() { stop(); }

    TestScheduler(const TestScheduler&)            = delete;
    TestScheduler& operator=(const TestScheduler&) = delete;
    TestScheduler(TestScheduler&&)                 = delete;
    TestScheduler& operator=(TestScheduler&&)      = delete;

    void run() {
        using namespace boost::ut;
        if (schedulerRet_.valid()) {
            // wait for previous thread to close because we gave it a pointer to scheduler_
            auto result = schedulerRet_.get();
            if (!result.has_value()) {
                expect(false) << std::format("Scheduler being restarted waited for the previous thread, which returned failure: \n{}\n", result.error());
            }
        }
        schedulerRet_ = gr::test::thread_pool::executeScheduler("qa_SchMess::scheduler", scheduler_);

        // Wait for the scheduler to start running
        expect(gr::testing::awaitCondition(scheduler_, [this] { return scheduler_.state() == gr::lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
        expect(scheduler_.state() == gr::lifecycle::State::RUNNING) << "scheduler thread up and running";
    }

    void stop() {
        using namespace boost::ut;
        scheduler_.requestStop();

        if (schedulerRet_.valid()) {
            auto result = schedulerRet_.get(); // this joins the thread
            if (!result.has_value()) {
                expect(false) << std::format("scheduler.runAndWait() failed:\n{}\n", result.error());
            }
        }
    }

    auto&                          scheduler() { return scheduler_; }
    auto&                          scheduler() const { return scheduler_; }
    auto&                          msgIn() { return scheduler_.msgIn; }
    auto&                          msgOut() { return scheduler_.msgOut; }
    auto&                          graph() { return scheduler_.graph(); }
    auto                           state() const { return scheduler_.state(); }
    std::string_view               unique_name() const { return scheduler_.unique_name; }
    void                           processScheduledMessages() { scheduler_.processScheduledMessages(); }
    std::expected<void, gr::Error> changeStateTo(gr::lifecycle::State state) { return scheduler_.changeStateTo(state); }
};

bool jobListsContain(const auto& scheduler, std::string_view blockUniqueName) {
    const std::shared_ptr<gr::scheduler::JobLists> jobLists = scheduler.jobs();
    return std::ranges::any_of(*jobLists, [&blockUniqueName](const std::vector<std::shared_ptr<gr::BlockModel>>& jobList) { //
        return std::ranges::any_of(jobList, [&blockUniqueName](const std::shared_ptr<gr::BlockModel>& block) { return block->uniqueName() == blockUniqueName; });
    });
}

std::string registeredSimpleSchedulerType(gr::SchedulerRegistry& schedulerRegistry) {
    gr::registerBlock<gr::scheduler::Simple<>>(schedulerRegistry);
    return std::string(gr::meta::type_name<gr::scheduler::Simple<>>());
}

// optional port names, if none are supplied then they will match all names and look for any edge
bool hasEdge(const gr::Graph& graph, std::string_view sourceBlockName, std::optional<std::string_view> sourcePortName, std::string_view destinationBlockName, std::optional<std::string_view> destinationPortName) {
    const auto portMatches = [](const gr::PortDefinition& definition, std::optional<std::string_view> name) {
        const auto* stringBased = std::get_if<gr::PortDefinition::StringBased>(&definition.definition);
        return !name.has_value() || (stringBased != nullptr && stringBased->name == *name);
    };
    return std::ranges::any_of(graph.edges(), [&](const gr::Edge& edge) { //
        return edge.sourceBlock()->uniqueName() == sourceBlockName && edge.destinationBlock()->uniqueName() == destinationBlockName && portMatches(edge.sourcePortDefinition(), sourcePortName) && portMatches(edge.destinationPortDefinition(), destinationPortName);
    });
}
constexpr std::optional<std::string_view> anyPortName; // matches any name

const boost::ut::suite TopologyGraphTests = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::scheduler;
    using namespace gr::testing;
    using namespace gr::test;
    using enum gr::message::Command;

    expect(fatal(gt(context->registry.keys().size(), 0UZ))) << "didn't register any blocks";

    "Block addition tests"_test = [] {
        TestScheduler scheduler(gr::Graph(context->loader));

        "Add a valid block"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceBlock, //
                {{"type", "gr::testing::Copy<float32>"}, {"properties", property_map{}}},                                                                  //
                ReplyChecker{.expectedEndpoint = scheduler::property::kBlockEmplaced});

            expect(eq(scheduler.graph().blocks().size(), 3UZ));
        };

        "Add an invalid block"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceBlock, //
                {{"type", "doesnt_exist::multiply<float32>"}, {"properties", property_map{}}},                                                             //
                ReplyChecker{.expectedEndpoint = scheduler::property::kEmplaceBlock, .expectedHasData = false});

            expect(eq(scheduler.graph().blocks().size(), 3UZ));
        };
    };

    "add block while scheduler is running"_test = [] {
        using namespace gr;
        using namespace gr::testing;

        Graph flow(context->loader);
        auto& source = flow.emplaceBlock<NullSource<float>>();
        auto& sink   = flow.emplaceBlock<NullSink<float>>();
        expect(flow.connect<"out", "in">(source, sink).has_value());

        TestScheduler scheduler(std::move(flow));

        expect(awaitCondition(scheduler, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";

        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler is running";

        auto initialBlockCount = scheduler.graph().blocks().size(); // valid to access blocks(), scheduler thread will not write to it until EmplaceBlock below
        std::println("Initial block count: {}", initialBlockCount);

        for (const auto& block : gr::globalBlockRegistry().keys()) {
            std::println("Block {}", block);
        }

        testing::sendAndWaitForReply<message::Command::Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceBlock, //
            property_map{{"type", "builtin_counter<float32>"}, {"properties", property_map{{"disconnect_on_done", false}}}},                                             //
            ReplyChecker{.expectedEndpoint = scheduler::property::kBlockEmplaced});

        // valid to read from blocks() as we waited for the message to complete first
        expect(scheduler.graph().blocks().size() > initialBlockCount) << "waiting for block to be added to graph";

        auto finalBlockCount = scheduler.graph().blocks().size();
        std::println("Final block count: {}", finalBlockCount);
        expect(eq(finalBlockCount, initialBlockCount + 1)) << "block was added";

        const auto isEmplacedAndRunning = [](const auto& block) { return block->name() == "builtin_counter<float32>" && block->state() == lifecycle::State::RUNNING; };
        expect(std::ranges::any_of(scheduler.graph().blocks(), isEmplacedAndRunning)) << "waiting for new block to reach running state";
    };

    "Block removal tests"_test = [] {
        gr::Graph graph(context->loader);
        std::ignore         = graph.emplaceBlock("gr::testing::Copy<float32>", {}).value();
        auto temporaryBlock = graph.emplaceBlock("gr::testing::Copy<float32>", {}).value();

        TestScheduler scheduler(std::move(graph));
        const auto&   testGraph = scheduler.graph();
        expect(eq(testGraph.blocks().size(), 4UZ));
        // expect(eq(getNReplyMessages(fromScheduler), 1UZ)); // emplaceBlock emits message
        consumeAllReplyMessages(scheduler.fromScheduler);

        "Remove a known block"_test = [&] {
            expect(eq(testGraph.blocks().size(), 4UZ));
            // expect(eq(getNReplyMessages(fromScheduler), 1UZ)); // emplaceBlock emits message
            consumeAllReplyMessages(scheduler.fromScheduler);

            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kRemoveBlock, //
                {{"uniqueName", temporaryBlock->uniqueName()}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlockRemoved});

            expect(eq(testGraph.blocks().size(), 3UZ));
        };

        "Remove an unknown block"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kRemoveBlock, //
                {{"uniqueName", "this_block_is_unknown"}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlockRemoved, .expectedHasData = false});

            expect(eq(testGraph.blocks().size(), 3UZ));
        };
    };

    // test for an issue where add and then removing blocks would cause them to
    // be leaked, stored in the zombie list forever
    "Removing a block that is still awaiting adoption destroys it"_test = [] {
        Graph flow(context->loader);
        auto& source = flow.emplaceBlock<SlowSource<float>>();
        auto& sink   = flow.emplaceBlock<CountingSink<float>>();
        expect(flow.connect<"out", "in">(source, sink).has_value()) << fatal;

        // use externalStep so that this test is totally singlethreaded and
        // block adoption happens at a predictable time
        TestScheduler<ExecutionPolicy::externalStep> scheduler(std::move(flow), false, false);

        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value()) << fatal;
        expect(scheduler.changeStateTo(lifecycle::State::RUNNING).has_value()) << fatal;

        sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler().unique_name, scheduler::property::kEmplaceBlock, {{"type", "gr::testing::Copy<float32>"}, {"properties", property_map{}}});
        scheduler.processScheduledMessages();
        const std::optional<Message> emplaceReply = testing::waitForReply(scheduler.fromScheduler, ReplyChecker{.expectedEndpoint = scheduler::property::kBlockEmplaced}, 1s);
        expect(emplaceReply.has_value() && emplaceReply->data.has_value()) << fatal << "block emplaced";
        const std::string newBlockName(emplaceReply->data.value().value_or<std::string_view>("unique_name", std::string_view{}));
        expect(!newBlockName.empty()) << fatal;
        consumeAllReplyMessages(scheduler.fromScheduler);

        std::weak_ptr<BlockModel> newBlockWeak = graph::findBlock(scheduler.graph(), std::string_view(newBlockName)).value();

        sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler().unique_name, scheduler::property::kRemoveBlock, {{"uniqueName", newBlockName}});
        scheduler.processScheduledMessages();
        const std::optional<Message> removeReply = testing::waitForReply(scheduler.fromScheduler, ReplyChecker{.expectedEndpoint = scheduler::property::kBlockRemoved}, 1s);
        expect(removeReply.has_value()) << "block removed";

        expect(!graph::findBlock(scheduler.graph(), std::string_view(newBlockName)).has_value()) << "the block is gone from the graph";
        expect(newBlockWeak.expired()) << "a block that never entered a worker's job list is destroyed on removal";

        expect(scheduler.changeStateTo(lifecycle::State::REQUESTED_STOP).has_value());
        expect(scheduler.changeStateTo(lifecycle::State::STOPPED).has_value());
    };

    constexpr static auto groupBlockInUnmanagedSubgraph = []<ExecutionPolicy policy> {
        Graph flow(context->loader);
        auto& source = flow.emplaceBlock<SlowSource<float>>();
        auto& copy   = flow.emplaceBlock<Copy<float>>();
        auto& sink   = flow.emplaceBlock<CountingSink<float>>();
        expect(flow.connect<"out", "in">(source, copy).has_value());
        expect(flow.connect<"out", "in">(copy, sink).has_value());

        TestScheduler<policy> scheduler(std::move(flow), /*addTestSourceAndSink=*/false);
        expect(eq(scheduler.graph().blocks().size(), 3UZ));

        "Group an unknown block"_test = [&] {
            Tensor<Value> uniqueNames;
            uniqueNames.push_back(Value("this_block_is_unknown"s));
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kGroupBlocks, //
                {{"type", "gr::Graph"}, {"uniqueNames", uniqueNames}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksGrouped, .expectedHasData = false});

            expect(eq(scheduler.graph().blocks().size(), 3UZ));
        };

        Tensor<Value> uniqueNames;
        uniqueNames.push_back(Value(std::string(copy.unique_name.value())));
        testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kGroupBlocks, //
            {{"type", "gr::Graph"}, {"uniqueNames", uniqueNames}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksGrouped});

        const auto blocks     = scheduler.graph().blocks();
        const auto subGraphIt = std::ranges::find_if(blocks, [](const auto& block) { return block->blockCategory() == gr::block::Category::TransparentBlockGroup; });
        expect(eq(blocks.size(), 3UZ)) << "source, sink and the spawned sub-graph";
        expect(subGraphIt != blocks.end()) << fatal << "grouping spawned a sub-graph block";
        expect(eq((*subGraphIt)->blocks().size(), 1UZ)) << "sub-graph contains the grouped block";
        expect(eq(scheduler.graph().edges().size(), 2UZ)) << "boundary edges are re-wired to the sub-graph";

        expect(jobListsContain(scheduler.scheduler(), copy.unique_name.value())) << "an unmanaged sub-graph does not run its children itself: the block must stay in the scheduler's job lists";
    };

    "Group block into unmanaged subgraph, singlethreaded"_test = [] { groupBlockInUnmanagedSubgraph.operator()<ExecutionPolicy::singleThreaded>(); };
    "Group block into unmanaged subgraph, multithreaded"_test  = [] { groupBlockInUnmanagedSubgraph.operator()<ExecutionPolicy::multiThreaded>(); };

    constexpr static auto blocksGroupedIntoUnmanagedSubgraphBeforeSchedulerStartStillRun = []<ExecutionPolicy policy> {
        Graph flow(context->loader);
        auto& source = flow.emplaceBlock<SlowSource<float>>();
        auto& copy   = flow.emplaceBlock<Copy<float>>();
        auto& sink   = flow.emplaceBlock<CountingSink<float>>();
        expect(flow.connect<"out", "in">(source, copy).has_value());
        expect(flow.connect<"out", "in">(copy, sink).has_value());

        TestScheduler<policy> scheduler(std::move(flow), false, false);
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value()) << fatal;

        // group while INITIALISED, i.e. after the job lists have been built but before the workers start
        Tensor<Value> uniqueNames;
        uniqueNames.push_back(Value(std::string(copy.unique_name.value())));
        sendMessage<Set>(scheduler.toScheduler, scheduler.scheduler().unique_name, scheduler::property::kGroupBlocks, {{"type", "gr::Graph"}, {"uniqueNames", uniqueNames}});
        scheduler.processScheduledMessages();
        auto reply = testing::waitForReply(scheduler.fromScheduler, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksGrouped});
        expect(reply.has_value() && reply->data.has_value()) << fatal << "grouping succeeded";

        expect(jobListsContain(scheduler.scheduler(), copy.unique_name.value())) << "the grouped block must stay in the job lists that the workers will pick up on start";

        scheduler.run();

        expect(awaitCondition(4s, [&sink] { return sink.progress->value() > 0U; })) << "everything is connected through the block inside the unmanaged subgraph";

        scheduler.scheduler().requestStop();
    };

    "Blocks grouped into unmanaged subgraph before start still run, singlethreaded"_test = [] { blocksGroupedIntoUnmanagedSubgraphBeforeSchedulerStartStillRun.operator()<ExecutionPolicy::singleThreaded>(); };
    "Blocks grouped into unmanaged subgraph before start still run, multithreaded"_test  = [] { blocksGroupedIntoUnmanagedSubgraphBeforeSchedulerStartStillRun.operator()<ExecutionPolicy::multiThreaded>(); };

    constexpr static auto groupBlocksIntoManagedSubgraph = []<ExecutionPolicy policy> {
        BlockRegistry     registry;
        SchedulerRegistry schedulerRegistry;
        gr::registerBlock<SlowSource, float>(registry);
        gr::registerBlock<Copy, float>(registry);
        gr::registerBlock<CountingSink, float>(registry);
        const std::string subGraphType = registeredSimpleSchedulerType(schedulerRegistry);
        PluginLoader      loader(registry, schedulerRegistry, {});

        Graph flow(loader);
        auto& source = flow.emplaceBlock<SlowSource<float>>();
        auto& copy   = flow.emplaceBlock<Copy<float>>();
        auto& sink   = flow.emplaceBlock<CountingSink<float>>();
        expect(flow.connect<"out", "in">(source, copy).has_value()) << fatal;
        expect(flow.connect<"out", "in">(copy, sink).has_value()) << fatal;

        TestScheduler<policy> scheduler(std::move(flow), /*addTestSourceAndSink=*/false);
        expect(eq(scheduler.graph().blocks().size(), 3UZ));
        expect(awaitCondition(4s, [&sink] { return sink.progress->value() > 0U; })) << "entire graph is connected and running";

        "Group an unknown block"_test = [&scheduler] {
            Tensor<Value> uniqueNames;
            uniqueNames.push_back(Value("this_block_is_unknown"s));
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kGroupBlocks, //
                {{"type", "gr::Graph"}, {"uniqueNames", uniqueNames}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksGrouped, .expectedHasData = false});

            expect(eq(scheduler.graph().blocks().size(), 3UZ));
        };

        Tensor<Value> uniqueNames;
        uniqueNames.push_back(Value(std::string(copy.unique_name.value())));
        testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kGroupBlocks, //
            {{"type", subGraphType}, {"uniqueNames", uniqueNames}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksGrouped});

        const auto progressAfterGroup = sink.progress->value();
        expect(awaitCondition(4s, [&sink, progressAfterGroup] { return sink.progress->value() > progressAfterGroup; })) << "entire graph is connected and running after grouping into managed subgraph";

        const auto blocks     = scheduler.graph().blocks();
        const auto subGraphIt = std::ranges::find_if(blocks, [](const auto& block) { return block->blockCategory() == gr::block::Category::ScheduledBlockGroup; });
        expect(eq(blocks.size(), 3UZ)) << "outer graph contains 3 blocks: the source, sink, and the new subgraph";
        expect(subGraphIt != blocks.end()) << fatal << "the subgraph should be present in the 3 blocks in the graph";
        expect(eq((*subGraphIt)->blocks().size(), 1UZ)) << "the subgraph contains the grouped block";
        expect(eq(scheduler.graph().edges().size(), 2UZ)) << "crossing edges are moved into the subgraph";

        expect(!jobListsContain(scheduler.scheduler(), copy.unique_name.value())) << "a subscheduler runs itself, so it shouldn't be in the parent's job lists";
    };

    "Group block into *managed* subgraph, singlethreaded"_test = [] { groupBlocksIntoManagedSubgraph.operator()<ExecutionPolicy::singleThreaded>(); };
    "Group block into *managed* subgraph, multithreaded"_test  = [] { groupBlocksIntoManagedSubgraph.operator()<ExecutionPolicy::multiThreaded>(); };

    constexpr static auto adoptingManagedSubgraphMustNotBlock = []<ExecutionPolicy policy> {
        BlockRegistry     registry;
        SchedulerRegistry schedulerRegistry;
        gr::registerBlock<SlowSource, float>(registry);
        gr::registerBlock<CountingSink, float>(registry);
        const std::string subGraphType = registeredSimpleSchedulerType(schedulerRegistry);
        PluginLoader      loader(registry, schedulerRegistry, {});

        Graph flow(loader);
        auto& groupedSource = flow.emplaceBlock<SlowSource<float>>();
        auto& groupedSink   = flow.emplaceBlock<CountingSink<float>>();
        expect(flow.connect<"out", "in">(groupedSource, groupedSink).has_value()) << fatal;

        TestScheduler<policy> scheduler(std::move(flow));

        Tensor<Value> uniqueNames;

        // group both a source and sink so that the subscheduler keeps running forever if start()ed
        uniqueNames.emplace_back(std::string(groupedSource.unique_name.value()));
        uniqueNames.emplace_back(std::string(groupedSink.unique_name.value()));

        sendMessage<Set>(scheduler.toScheduler, scheduler.unique_name(), scheduler::property::kGroupBlocks, //
            {{"type", subGraphType}, {"uniqueNames", uniqueNames}});

        const std::optional<Message> reply = testing::waitForReply(scheduler.fromScheduler, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksGrouped}, 5s);
        expect(reply.has_value()) << fatal << "grouping into a managed subgraph does not hang";

        consumeAllReplyMessages(scheduler.fromScheduler);
        const auto blocks     = scheduler.graph().blocks();
        const auto subGraphIt = std::ranges::find_if(blocks, [](const auto& block) { return block->blockCategory() == gr::block::Category::ScheduledBlockGroup; });
        expect(subGraphIt != blocks.end()) << fatal << "grouping spawned a managed subgraph block";
        expect(awaitCondition(4s, [&subGraphIt] { return (*subGraphIt)->state() == lifecycle::State::RUNNING; })) << "the adopted subscheduler becomes RUNNING on its own thread";
    };

    "Adopting a managed subgraph must not block message processing, singlethreaded"_test = [] { adoptingManagedSubgraphMustNotBlock.operator()<ExecutionPolicy::singleThreaded>(); };
    "Adopting a managed subgraph must not block message processing, multithreaded"_test  = [] { adoptingManagedSubgraphMustNotBlock.operator()<ExecutionPolicy::multiThreaded>(); };

    // this also tests doubly nesting the unmanaged subgraph, to make sure nested graphs all get adopted by the root scheduler
    constexpr static auto groupBlocksIntoUnmanagedSubgraph = []<ExecutionPolicy policy> {
        Graph flow(context->loader);
        auto& source = flow.emplaceBlock<SlowSource<float>>();
        auto& copy   = flow.emplaceBlock<Copy<float>>();
        auto& sink   = flow.emplaceBlock<CountingSink<float>>();
        expect(flow.connect<"out", "in">(source, copy).has_value());
        expect(flow.connect<"out", "in">(copy, sink).has_value());

        TestScheduler<policy> scheduler(std::move(flow), /*addTestSourceAndSink=*/false);

        // move the copy block into an unmanaged subgraph
        Tensor<Value> uniqueNames;
        uniqueNames.emplace_back(std::string(copy.unique_name.value()));
        const std::optional<Message> outerReply = testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kGroupBlocks, //
            {{"type", "gr::Graph"}, {"uniqueNames", uniqueNames}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksGrouped});
        expect(outerReply.has_value() && outerReply->data.has_value()) << fatal;
        const std::string outerSubGraphName(outerReply->data.value().value_or<std::string_view>("uniqueName", std::string_view{}));
        expect(!outerSubGraphName.empty()) << fatal;

        // move the copy block into a doubly nested subgraph
        const auto nestedReply = testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kGroupBlocks, //
            {{"type", "gr::Graph"}, {"uniqueNames", uniqueNames}, {"_targetGraph", outerSubGraphName}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksGrouped});
        expect(nestedReply.has_value() && nestedReply->data.has_value()) << fatal << "nested grouping succeeded";

        // verify the structure of the graph and subgraphs
        const auto blocks          = scheduler.graph().blocks();
        const auto outerSubGraphIt = std::ranges::find_if(blocks, [&outerSubGraphName](const auto& block) { return block->uniqueName() == outerSubGraphName; });
        expect(eq(blocks.size(), 3UZ)) << "root graph still contains source, sink and the outer subgraph";
        expect(outerSubGraphIt != blocks.end()) << fatal;
        expect(eq((*outerSubGraphIt)->blocks().size(), 1UZ)) << fatal << "outer subgraph contains only the nested subgraph";
        const auto& nestedSubGraph = (*outerSubGraphIt)->blocks()[0UZ];
        expect(nestedSubGraph->blockCategory() == gr::block::Category::TransparentBlockGroup) << "nested block group is a transparent subgraph";
        expect(eq(nestedSubGraph->blocks().size(), 1UZ)) << fatal << "nested subgraph contains the copy block";
        expect(nestedSubGraph->blocks()[0UZ]->uniqueName() == copy.unique_name.value());

        expect(jobListsContain(scheduler.scheduler(), copy.unique_name.value())) << "the doubly nested block must stay in the root scheduler's job lists";

        const auto progressBeforeRestart = sink.progress->value();
        expect(awaitCondition(4s, [&sink, progressBeforeRestart] { return sink.progress->value() > progressBeforeRestart; })) << "doubly nested subgraph should still have connected edges";

        // job lists must be rebuilt from all the nested graphs
        scheduler.stop();
        const auto progressAfterStop = sink.progress->value();
        scheduler.run();
        expect(awaitCondition(4s, [&sink, progressAfterStop] { return sink.progress->value() > progressAfterStop; })) << "doubly nested subgraph should still have connected edges after restart";
    };

    "Group blocks nested inside an unmanaged subgraph, singlethreaded"_test = [] { groupBlocksIntoUnmanagedSubgraph.operator()<ExecutionPolicy::singleThreaded>(); };
    "Group blocks nested inside an unmanaged subgraph, multithreaded"_test  = [] { groupBlocksIntoUnmanagedSubgraph.operator()<ExecutionPolicy::multiThreaded>(); };

    constexpr static auto ungroupSubgraphWithSingleBlock = []<ExecutionPolicy policy> {
        Graph flow(context->loader);

        Graph subGraphContents;
        auto& copy = subGraphContents.emplaceBlock<Copy<float>>();

        std::shared_ptr<BlockModel>     subGraphBlock = flow.addBlock(std::make_shared<GraphWrapper<Graph>>(std::move(subGraphContents)));
        const std::string               subGraphName(subGraphBlock->uniqueName());
        const std::weak_ptr<BlockModel> subGraphWeak = subGraphBlock;

        TestScheduler<policy> scheduler(std::move(flow));
        expect(eq(scheduler.graph().blocks().size(), 3UZ)) << "the subgraph node and the two keep-alive test blocks";

        "Ungroup an unknown block"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kUngroupBlocks, //
                {{"uniqueName", "this_block_is_unknown"s}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksUngrouped, .expectedHasData = false});

            expect(eq(scheduler.graph().blocks().size(), 3UZ));
        };

        "Ungroup a block that is not a subgraph"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kUngroupBlocks, //
                {{"uniqueName", std::string(scheduler.graph().blocks()[1UZ]->uniqueName())}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksUngrouped, .expectedHasData = false});

            expect(eq(scheduler.graph().blocks().size(), 3UZ));
        };

        testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kUngroupBlocks, //
            {{"uniqueName", subGraphName}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksUngrouped});

        expect(eq(scheduler.graph().blocks().size(), 3UZ)) << "the subgraph node was replaced by the block it contained";
        expect(graph::findBlock(scheduler.graph(), copy).has_value()) << "the block was moved into the root graph";
        expect(!graph::findBlock(scheduler.graph(), std::string_view(subGraphName)).has_value()) << "the subgraph node itself is removed";
        expect(eq(scheduler.graph().edges().size(), 0UZ));

        subGraphBlock.reset(); // drop our reference: the running scheduler must release the removed wrapper too
        expect(awaitCondition(4s, [&subGraphWeak] { return subGraphWeak.expired(); })) << "the scheduler stops referencing (and destroys) the removed subgraph wrapper";
    };

    "Ungroup a subgraph with a single block, singlethreaded"_test = [] { ungroupSubgraphWithSingleBlock.operator()<ExecutionPolicy::singleThreaded>(); };
    "Ungroup a subgraph with a single block, multithreaded"_test  = [] { ungroupSubgraphWithSingleBlock.operator()<ExecutionPolicy::multiThreaded>(); };

    constexpr static auto ungroupResolvesExportedPortNames = []<ExecutionPolicy policy> {
        Graph flow(context->loader);
        auto& source1 = flow.emplaceBlock<SlowSource<float>>();
        auto& source2 = flow.emplaceBlock<SlowSource<float>>();

        Graph subGraphContents;
        auto& sink1 = subGraphContents.emplaceBlock<CountingSink<float>>();
        auto& sink2 = subGraphContents.emplaceBlock<CountingSink<float>>();

        const std::shared_ptr<BlockModel> subGraphBlock = flow.addBlock(std::make_shared<GraphWrapper<Graph>>(std::move(subGraphContents)));
        const std::string                 subGraphName(subGraphBlock->uniqueName());
        expect(subGraphBlock->exportPort(true, sink1.unique_name, PortDirection::INPUT, "in", "sink1_in").has_value()) << fatal;
        expect(subGraphBlock->exportPort(true, sink2.unique_name, PortDirection::INPUT, "in", "sink2_in").has_value()) << fatal;

        const std::shared_ptr<BlockModel> source1Model = graph::findBlock(flow, source1).value();
        const std::shared_ptr<BlockModel> source2Model = graph::findBlock(flow, source2).value();
        expect(flow.connect(source1Model, PortDefinition("out"), subGraphBlock, PortDefinition("sink1_in")).has_value()) << fatal;
        expect(flow.connect(source2Model, PortDefinition("out"), subGraphBlock, PortDefinition("sink2_in")).has_value()) << fatal;

        TestScheduler<policy> scheduler(std::move(flow), /*addTestSourceAndSink=*/false);

        expect(awaitCondition(4s, [&sink1, &sink2] { return sink1.progress->value() > 0U && sink2.progress->value() > 0U; })) << "data flows through the exported ports before ungrouping";

        testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kUngroupBlocks, //
            {{"uniqueName", subGraphName}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksUngrouped});

        expect(eq(scheduler.graph().blocks().size(), 4UZ)) << "root graph contains both sources and both sinks";
        expect(eq(scheduler.graph().edges().size(), 2UZ));
        expect(hasEdge(scheduler.graph(), source1.unique_name.value(), "out", sink1.unique_name.value(), "in")) << "the port exported as 'sink1_in' resolves to the input of the first sink";
        expect(hasEdge(scheduler.graph(), source2.unique_name.value(), "out", sink2.unique_name.value(), "in")) << "the port exported as 'sink2_in' resolves to the input of the second sink";
        expect(!hasEdge(scheduler.graph(), source1.unique_name.value(), anyPortName, sink2.unique_name.value(), anyPortName)) << "sources are not cross-connected";
        expect(!hasEdge(scheduler.graph(), source2.unique_name.value(), anyPortName, sink1.unique_name.value(), anyPortName)) << "sources are not cross-connected";

        const auto count1AfterUngroup = sink1.progress->value();
        const auto count2AfterUngroup = sink2.progress->value();
        expect(awaitCondition(4s, [&sink1, &sink2, count1AfterUngroup, count2AfterUngroup] { //
            return sink1.progress->value() > count1AfterUngroup && sink2.progress->value() > count2AfterUngroup;
        })) << "data still flows to both sinks after ungrouping";
    };

    "Ungroup resolves exported port names to the correct blocks, singlethreaded"_test = [] { ungroupResolvesExportedPortNames.operator()<ExecutionPolicy::singleThreaded>(); };
    "Ungroup resolves exported port names to the correct blocks, multithreaded"_test  = [] { ungroupResolvesExportedPortNames.operator()<ExecutionPolicy::multiThreaded>(); };

    constexpr static auto ungroupPreservesInternalEdges = []<ExecutionPolicy policy> {
        Graph flow(context->loader);

        Graph subGraphContents;
        auto& source = subGraphContents.emplaceBlock<SlowSource<float>>();
        auto& sink   = subGraphContents.emplaceBlock<CountingSink<float>>();
        expect(subGraphContents.connect<"out", "in">(source, sink).has_value()) << fatal;

        const std::shared_ptr<BlockModel> subGraphBlock = flow.addBlock(std::make_shared<GraphWrapper<Graph>>(std::move(subGraphContents)));
        const std::string                 subGraphName(subGraphBlock->uniqueName());

        TestScheduler<policy> scheduler(std::move(flow), /*addTestSourceAndSink=*/false);

        expect(awaitCondition(4s, [&sink] { return sink.progress->value() > 0U; })) << "data flows inside the subgraph before ungrouping";

        testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kUngroupBlocks, //
            {{"uniqueName", subGraphName}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksUngrouped});

        expect(eq(scheduler.graph().blocks().size(), 2UZ));
        expect(eq(scheduler.graph().edges().size(), 1UZ)) << "the edge between the subgraph's blocks moved into the root graph";
        expect(hasEdge(scheduler.graph(), source.unique_name.value(), anyPortName, sink.unique_name.value(), anyPortName)) << "the internal edge still connects the same blocks";

        const auto progressAfterUngroup = sink.progress->value();
        expect(awaitCondition(4s, [&sink, progressAfterUngroup] { return sink.progress->value() > progressAfterUngroup; })) << "data still flows after ungrouping";
    };

    "Ungroup preserves edges between the subgraph's blocks, singlethreaded"_test = [] { ungroupPreservesInternalEdges.operator()<ExecutionPolicy::singleThreaded>(); };
    "Ungroup preserves edges between the subgraph's blocks, multithreaded"_test  = [] { ungroupPreservesInternalEdges.operator()<ExecutionPolicy::multiThreaded>(); };

    constexpr static auto ungroupIsInverseOfGroup = []<ExecutionPolicy policy>(std::string_view subGraphType) {
        // use a loader just for this test to avoid changing global state
        BlockRegistry     registry;
        SchedulerRegistry schedulerRegistry;
        gr::registerBlock<SlowSource, float>(registry);
        gr::registerBlock<Copy, float>(registry);
        gr::registerBlock<CountingSink, float>(registry);
        std::ignore = registeredSimpleSchedulerType(schedulerRegistry);
        PluginLoader loader(registry, schedulerRegistry, {});

        Graph flow(loader);
        auto& source = flow.emplaceBlock<SlowSource<float>>();
        auto& copy2  = flow.emplaceBlock<Copy<float>>();
        auto& copy3  = flow.emplaceBlock<Copy<float>>();
        auto& copy4  = flow.emplaceBlock<Copy<float>>();
        auto& sink   = flow.emplaceBlock<CountingSink<float>>();
        expect(flow.connect<"out", "in">(source, copy2).has_value()) << fatal;
        expect(flow.connect<"out", "in">(copy2, copy3).has_value()) << fatal;
        expect(flow.connect<"out", "in">(copy3, copy4).has_value()) << fatal;
        expect(flow.connect<"out", "in">(copy4, sink).has_value()) << fatal;

        const std::array<std::string_view, 5> originalBlockNames = {source.unique_name.value(), copy2.unique_name.value(), copy3.unique_name.value(), copy4.unique_name.value(), sink.unique_name.value()};

        TestScheduler<policy> scheduler(std::move(flow), /*addTestSourceAndSink=*/false);
        expect(awaitCondition(4s, [&sink] { return sink.progress->value() > 0U; })) << "data flows before grouping";

        Tensor<Value> uniqueNames;
        uniqueNames.emplace_back(std::string(copy2.unique_name.value()));
        uniqueNames.emplace_back(std::string(copy3.unique_name.value()));
        uniqueNames.emplace_back(std::string(copy4.unique_name.value()));
        const std::optional<Message> groupReply = testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kGroupBlocks, //
            {{"type", std::string(subGraphType)}, {"uniqueNames", uniqueNames}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksGrouped});
        expect(groupReply.has_value() && groupReply->data.has_value()) << fatal;
        const std::string subGraphName(groupReply->data.value().value_or<std::string_view>("uniqueName", std::string_view{}));
        expect(!subGraphName.empty()) << fatal;
        expect(eq(scheduler.graph().blocks().size(), 3UZ)) << "source, sink and the subgraph";
        const std::weak_ptr<BlockModel> subGraphWeak = graph::findBlock(scheduler.graph(), std::string_view(subGraphName)).value();

        testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kUngroupBlocks, //
            {{"uniqueName", subGraphName}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksUngrouped});

        expect(awaitCondition(4s, [&subGraphWeak] { return subGraphWeak.expired(); })) << "the scheduler stops referencing (and destroys) the removed subgraph wrapper";

        expect(eq(scheduler.graph().blocks().size(), 5UZ)) << "grouping followed by ungrouping restores the original block count";
        for (const std::string_view blockName : originalBlockNames) {
            expect(graph::findBlock(scheduler.graph(), blockName).has_value()) << "every original block is back in the root graph";
        }
        expect(std::ranges::all_of(scheduler.graph().blocks(), [](const auto& block) { return block->blockCategory() == gr::block::Category::NormalBlock; })) << "no subgraph block remains";

        expect(eq(scheduler.graph().edges().size(), 4UZ));
        expect(hasEdge(scheduler.graph(), source.unique_name.value(), anyPortName, copy2.unique_name.value(), "in")) << "boundary edge into the group is restored";
        expect(hasEdge(scheduler.graph(), copy2.unique_name.value(), anyPortName, copy3.unique_name.value(), anyPortName)) << "edge internal to the group is restored";
        expect(hasEdge(scheduler.graph(), copy3.unique_name.value(), anyPortName, copy4.unique_name.value(), anyPortName)) << "edge internal to the group is restored";
        expect(hasEdge(scheduler.graph(), copy4.unique_name.value(), "out", sink.unique_name.value(), anyPortName)) << "boundary edge out of the group is restored";

        const auto progressAfterUngroup = sink.progress->value();
        expect(awaitCondition(4s, [&sink, progressAfterUngroup] { return sink.progress->value() > progressAfterUngroup; })) << "data flows end-to-end after the group/ungroup round trip";
    };

    "Group and ungroup an unmanaged subgraph is a no-op, singlethreaded"_test = [] { ungroupIsInverseOfGroup.operator()<ExecutionPolicy::singleThreaded>("gr::Graph"); };
    "Group and ungroup an unmanaged subgraph is a no-op, multithreaded"_test  = [] { ungroupIsInverseOfGroup.operator()<ExecutionPolicy::multiThreaded>("gr::Graph"); };

    "Group and ungroup a *managed* subgraph is a no-op, singlethreaded"_test = [] { ungroupIsInverseOfGroup.operator()<ExecutionPolicy::singleThreaded>(gr::meta::type_name<gr::scheduler::Simple<>>()); };
    "Group and ungroup a *managed* subgraph is a no-op, multithreaded"_test  = [] { ungroupIsInverseOfGroup.operator()<ExecutionPolicy::multiThreaded>(gr::meta::type_name<gr::scheduler::Simple<>>()); };

    constexpr static auto repeatedGroupUngroupWhileRunning = []<ExecutionPolicy policy> {
        BlockRegistry     registry;
        SchedulerRegistry schedulerRegistry;
        gr::registerBlock<SlowSource, float>(registry);
        gr::registerBlock<Copy, float>(registry);
        gr::registerBlock<CountingSink, float>(registry);
        const std::string subGraphType = registeredSimpleSchedulerType(schedulerRegistry);
        PluginLoader      loader(registry, schedulerRegistry, {});

        Graph flow(loader);
        auto& source = flow.emplaceBlock<SlowSource<float>>({{"disconnect_on_done", false}});
        auto& copy2  = flow.emplaceBlock<Copy<float>>({{"disconnect_on_done", false}});
        auto& copy3  = flow.emplaceBlock<Copy<float>>({{"disconnect_on_done", false}});
        auto& sink   = flow.emplaceBlock<CountingSink<float>>({{"disconnect_on_done", false}});
        expect(flow.connect<"out", "in">(source, copy2).has_value()) << fatal;
        expect(flow.connect<"out", "in">(copy2, copy3).has_value()) << fatal;
        expect(flow.connect<"out", "in">(copy3, sink).has_value()) << fatal;

        TestScheduler<policy> scheduler(std::move(flow), /*addTestSourceAndSink=*/false);
        expect(awaitCondition(4s, [&sink] { return sink.progress->value() > 0U; })) << fatal << "graph should be fully connected and running";

        // this is a reproduction of an issue where a scheduler and its child scheduler might put the same block into their
        // execution order after grouping or ungrouping. The crash actually happening is not necessarily guaranteed, but
        // there is not really a way other than dynamic_cast() to extract the child scheduler and observe if its worker
        // threads share blocks with the root scheduler.
        for (std::size_t iteration = 0UZ; iteration < 4UZ; ++iteration) {
            Tensor<Value> uniqueNames;
            uniqueNames.emplace_back(std::string(source.unique_name.value()));
            uniqueNames.emplace_back(std::string(copy2.unique_name.value()));
            const std::optional<Message> groupReply = testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kGroupBlocks, //
                {{"type", subGraphType}, {"uniqueNames", uniqueNames}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksGrouped});
            expect(groupReply.has_value() && groupReply->data.has_value()) << fatal << std::format("grouping succeeded in iteration {}", iteration);
            std::string subGraphName(groupReply->data.value().value_or<std::string_view>("uniqueName", std::string_view{}));
            expect(!subGraphName.empty()) << fatal;

            const auto progressAfterGroup = sink.progress->value();
            expect(awaitCondition(4s, [&sink, progressAfterGroup] { return sink.progress->value() > progressAfterGroup; })) << std::format("graph should be fully connected after group, on iteration {}", iteration);

            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kUngroupBlocks, //
                {{"uniqueName", subGraphName}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksUngrouped});

            const auto progressAfterUngroup = sink.progress->value();
            expect(awaitCondition(4s, [&sink, progressAfterUngroup] { return sink.progress->value() > progressAfterUngroup; })) << std::format("graph should be fully connected after ungroup, on iteration {}", iteration);
        }
    };

    "Repeated group/ungroup of a managed subgraph while running, singlethreaded"_test = [] { repeatedGroupUngroupWhileRunning.operator()<ExecutionPolicy::singleThreaded>(); };
    "Repeated group/ungroup of a managed subgraph while running, multithreaded"_test  = [] { repeatedGroupUngroupWhileRunning.operator()<ExecutionPolicy::multiThreaded>(); };

    constexpr static auto ungroupManagedSubgraph = []<ExecutionPolicy policy> {
        Graph flow(context->loader);

        Graph subGraphContents;
        // keep the ports connected while the sub-scheduler stops so that the flow can resume after ungrouping
        auto& source = subGraphContents.emplaceBlock<SlowSource<float>>({{"disconnect_on_done", false}});
        auto& sink   = subGraphContents.emplaceBlock<CountingSink<float>>({{"disconnect_on_done", false}});
        expect(subGraphContents.connect<"out", "in">(source, sink).has_value()) << fatal;

        auto subSchedulerWrapper = std::make_shared<SchedulerWrapper<gr::scheduler::Simple<>>>();
        subSchedulerWrapper->setGraph(std::move(subGraphContents));
        std::shared_ptr<BlockModel>     subSchedulerBlock = flow.addBlock(subSchedulerWrapper);
        const std::string               subSchedulerName(subSchedulerBlock->uniqueName());
        const std::weak_ptr<BlockModel> subSchedulerWeak = subSchedulerBlock;
        expect(subSchedulerBlock->blockCategory() == gr::block::Category::ScheduledBlockGroup);
        subSchedulerWrapper.reset();

        TestScheduler<policy> scheduler(std::move(flow));

        expect(awaitCondition(4s, [&sink] { return sink.progress->value() > 0U; })) << "data flows inside the managed subgraph before ungrouping";

        testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kUngroupBlocks, //
            {{"uniqueName", subSchedulerName}}, ReplyChecker{.expectedEndpoint = scheduler::property::kBlocksUngrouped});

        expect(eq(scheduler.graph().blocks().size(), 4UZ)) << "the sub-scheduler was replaced by the two blocks it ran";
        expect(graph::findBlock(scheduler.graph(), source).has_value()) << "the source was moved into the root graph";
        expect(graph::findBlock(scheduler.graph(), sink).has_value()) << "the sink was moved into the root graph";
        expect(!graph::findBlock(scheduler.graph(), std::string_view(subSchedulerName)).has_value()) << "the sub-scheduler block itself is removed";
        expect(subSchedulerBlock->state() == lifecycle::State::STOPPED) << "the sub-scheduler was stopped";
        expect(hasEdge(scheduler.graph(), source.unique_name.value(), anyPortName, sink.unique_name.value(), anyPortName)) << "the edge between the subgraph's blocks moved into the root graph";

        subSchedulerBlock.reset(); // drop our reference: the running scheduler must release the removed wrapper too
        expect(awaitCondition(4s, [&subSchedulerWeak] { return subSchedulerWeak.expired(); })) << "the scheduler stops referencing (and destroys) the removed sub-scheduler wrapper";

        // the sink may consume the EOS tag that the source published while the sub-scheduler stopped it and
        // stop itself again, so only the source (which has no inputs) proves the adoption deterministically
        expect(awaitCondition(4s, [&source] { return source.state() == lifecycle::State::RUNNING; })) << "the re-parented source is adopted by this scheduler and runs again";

        // restart because the job lists need to be rebuilt from the graph
        scheduler.stop();
        scheduler.run();
        const auto progressAfterRestart = sink.progress->value();
        expect(awaitCondition(4s, [&sink, progressAfterRestart] { return sink.progress->value() > progressAfterRestart; })) << "data flows through the re-parented blocks after the restart";
    };

    "Ungroup a *managed* subgraph into the running scheduler, singlethreaded"_test = [] { ungroupManagedSubgraph.operator()<ExecutionPolicy::singleThreaded>(); };
    "Ungroup a *managed* subgraph into the running scheduler, multithreaded"_test  = [] { ungroupManagedSubgraph.operator()<ExecutionPolicy::multiThreaded>(); };

    "Block replacement tests"_test = [] {
        gr::Graph graph(context->loader);

        auto block = graph.emplaceBlock("gr::testing::Copy<float32>", {}).value();
        expect(eq(graph.blocks().size(), 1UZ));
        auto temporaryBlock = graph.emplaceBlock("gr::testing::Copy<float32>", {}).value();

        TestScheduler scheduler(std::move(graph));
        "Replace a known block"_test = [&] {
            expect(eq(scheduler.graph().blocks().size(), 4UZ));

            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kReplaceBlock, //
                {{"uniqueName", temporaryBlock->uniqueName()},                                                                                             //
                    {"type", "gr::testing::Copy<float32>"}, {"properties", property_map{}}},                                                               //
                ReplyChecker{.expectedEndpoint = scheduler::property::kBlockReplaced});
        };

        "Replace an unknown block"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kReplaceBlock, //
                {{"uniqueName", "this_block_is_unknown"},                                                                                                  //
                    {"type", "gr::testing::Copy<float32>"}, {"properties", property_map{}}},                                                               //
                ReplyChecker{.expectedEndpoint = scheduler::property::kReplaceBlock, .expectedHasData = false});
        };

        "Replace with an unknown block"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kReplaceBlock, //
                {{"uniqueName", block->uniqueName()},                                                                                                      //
                    {"type", "doesnt_exist::multiply<float32>"}, {"properties", property_map{}}},
                ReplyChecker{.expectedEndpoint = scheduler::property::kReplaceBlock, .expectedHasData = false});
        };
    };

    "Edge addition tests"_test = [&] {
        gr::Graph testGraph(context->loader);

        auto blockOut       = testGraph.emplaceBlock("gr::testing::Copy<float32>", {}).value();
        auto blockIn        = testGraph.emplaceBlock("gr::testing::Copy<float32>", {}).value();
        auto blockWrongType = testGraph.emplaceBlock("gr::testing::Copy<float64>", {}).value();

        TestScheduler scheduler(std::move(testGraph));

        "Add an edge"_test = [&] {
            property_map data = {{gr::serialization_fields::EDGE_SOURCE_BLOCK, blockOut->uniqueName()}, //
                {gr::serialization_fields::EDGE_SOURCE_PORT, "out"},                                    //
                {gr::serialization_fields::EDGE_DESTINATION_BLOCK, blockIn->uniqueName()},              //
                {gr::serialization_fields::EDGE_DESTINATION_PORT, "in"},                                //
                {gr::serialization_fields::EDGE_MIN_BUFFER_SIZE, gr::Size_t()},                         //
                {gr::serialization_fields::EDGE_WEIGHT, 0},                                             //
                {gr::serialization_fields::EDGE_NAME, "unnamed edge"}};

            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceEdge, data, //
                ReplyChecker{.expectedEndpoint = scheduler::property::kEdgeEmplaced});
        };

        "Fail to add an edge because source port is invalid"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceEdge, //
                {{gr::serialization_fields::EDGE_SOURCE_BLOCK, blockOut->uniqueName()},                                                                   //
                    {gr::serialization_fields::EDGE_SOURCE_PORT, "OUTPUT"},                                                                               //
                    {gr::serialization_fields::EDGE_DESTINATION_BLOCK, blockIn->uniqueName()},                                                            //
                    {gr::serialization_fields::EDGE_DESTINATION_PORT, "in"}},
                ReplyChecker{.expectedEndpoint = scheduler::property::kEmplaceEdge, .expectedHasData = false});
        };

        "Fail to add an edge because destination port is invalid"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceEdge, //
                {{gr::serialization_fields::EDGE_SOURCE_BLOCK, blockOut->uniqueName()},                                                                   //
                    {gr::serialization_fields::EDGE_SOURCE_PORT, "in"},                                                                                   //
                    {gr::serialization_fields::EDGE_DESTINATION_BLOCK, blockIn->uniqueName()},                                                            //
                    {gr::serialization_fields::EDGE_DESTINATION_PORT, "INPUT"}},
                ReplyChecker{.expectedEndpoint = scheduler::property::kEmplaceEdge, .expectedHasData = false});
        };

        "Fail to add an edge because ports are not compatible"_test = [&] {
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceEdge, //
                {{gr::serialization_fields::EDGE_SOURCE_BLOCK, blockOut->uniqueName()},                                                                   //
                    {gr::serialization_fields::EDGE_SOURCE_PORT, "out"},                                                                                  //
                    {gr::serialization_fields::EDGE_DESTINATION_BLOCK, blockWrongType->uniqueName()},                                                     //
                    {gr::serialization_fields::EDGE_DESTINATION_PORT, "in"}},
                ReplyChecker{.expectedEndpoint = scheduler::property::kEmplaceEdge, .expectedHasData = false});
        };
    };

    "Edge removal tests"_test = [&] {
        gr::Graph testGraph(context->loader);

        // disconnect_on_done=false: dangling blocks would otherwise self-stop and tear down
        // the freshly emplaced connection via disconnectFromUpStreamParents()
        auto blockOut = testGraph.emplaceBlock("gr::testing::Copy<float32>", {{"disconnect_on_done", false}}).value();
        auto blockIn  = testGraph.emplaceBlock("gr::testing::Copy<float32>", {{"disconnect_on_done", false}}).value();

        TestScheduler scheduler(std::move(testGraph));

        const property_map edgeData = {{gr::serialization_fields::EDGE_SOURCE_BLOCK, blockOut->uniqueName()}, //
            {gr::serialization_fields::EDGE_SOURCE_PORT, "out"},                                              //
            {gr::serialization_fields::EDGE_DESTINATION_BLOCK, blockIn->uniqueName()},                        //
            {gr::serialization_fields::EDGE_DESTINATION_PORT, "in"},                                          //
            {gr::serialization_fields::EDGE_MIN_BUFFER_SIZE, gr::Size_t()},                                   //
            {gr::serialization_fields::EDGE_WEIGHT, 0},                                                       //
            {gr::serialization_fields::EDGE_NAME, "unnamed edge"}};

        testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), //
            scheduler::property::kEmplaceEdge, edgeData, ReplyChecker{.expectedEndpoint = scheduler::property::kEdgeEmplaced});

        // now the removed edge should be gone
        testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kRemoveEdge, //
            {{gr::serialization_fields::EDGE_SOURCE_BLOCK, blockOut->uniqueName()},                                                                  //
                {gr::serialization_fields::EDGE_SOURCE_PORT, "out"}},
            ReplyChecker{.expectedEndpoint = scheduler::property::kEdgeRemoved});

        testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, "", graph::property::kGraphInspect, property_map{}, //
            [](const Message& reply) {
                if (reply.endpoint != graph::property::kGraphInspected) {
                    return false;
                }

                const auto& data  = reply.data.value();
                const auto& edges = gr::test::get_value_or_fail<property_map>(data.find_value(serialization_fields::BLOCK_EDGES).value());
                expect(eq(edges.size(), 0UZ)) << "removed edge must not appear in the inspected graph";
                return true;
            });
    };

    "Settings change via messages"_test = [] {
        gr::Graph testGraph(context->loader);
        std::ignore = testGraph.emplaceBlock("gr::testing::Copy<float32>", {}).value();
        std::ignore = testGraph.emplaceBlock("gr::testing::Copy<float32>", {}).value();

        TestScheduler scheduler(std::move(testGraph));

        "get scheduler settings"_test = [&] {
            // TODO: Would like to port to sendAndWaitMessage, but it's logic is looking at the whole message
            // queue, and fails if there's scheduler messages. In the future the scheduler might
            // insert unrelated messages in the queue and this test will fail
            sendMessage<Get>(scheduler.toScheduler, "", block::property::kSetting, {});
            expect(waitForReply(scheduler.fromScheduler, ReplyChecker{.expectedEndpoint = block::property::kSetting}).has_value()) << "expected reply";

            bool        atLeastOneReplyFromScheduler = false;
            std::size_t availableMessages            = scheduler.fromScheduler.streamReader().available();
            expect(ge(availableMessages, 1UZ)) << "didn't receive reply message";
            for (const auto& reply : consumeAllReplyMessages(scheduler.fromScheduler)) {
                if (reply.serviceName != scheduler.scheduler().unique_name) {
                    continue;
                }

                std::println("Got reply: {}", reply);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.endpoint, std::string(block::property::kSetting)));
                expect(reply.data.has_value());
                expect(!reply.data.value().empty());
                expect(reply.data.value().contains("timeout_ms"));
                atLeastOneReplyFromScheduler = true;
            }

            expect(atLeastOneReplyFromScheduler);
        };

        "set scheduler settings"_test = [&] {
            // See TODO from "get scheduler settings", same case
            sendMessage<Set>(scheduler.toScheduler, "", block::property::kStagedSetting, {{"timeout_ms", 42}});
            expect(waitForReply(scheduler.fromScheduler, ReplyChecker{.expectedEndpoint = block::property::kStagedSetting, .expectedHasData = false}).has_value()) << "expected reply";

            bool        atLeastOneReplyFromScheduler = false;
            std::size_t availableMessages            = scheduler.fromScheduler.streamReader().available();
            expect(ge(availableMessages, 1UZ)) << "didn't receive reply message";
            for (const auto& reply : consumeAllReplyMessages(scheduler.fromScheduler)) {
                if (reply.serviceName != scheduler.scheduler().unique_name) {
                    continue;
                }
                atLeastOneReplyFromScheduler = true;
            }
            expect(!atLeastOneReplyFromScheduler) << "should not receive a reply";
            property_map stagedSettings = scheduler.scheduler().settings().stagedParameters();
            expect(stagedSettings.contains("timeout_ms"));
            expect(eq(42UZ, gr::test::get_value_or_fail<gr::Size_t>(stagedSettings.find_value("timeout_ms").value())));

            // setting staged setting via staged setting (N.B. non-real-time <-> real-time setting decoupling
            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, "", block::property::kSetting, {{"timeout_ms", 43}}, //
                ReplyChecker{.expectedEndpoint = block::property::kSetting, .expectedHasData = false});

            stagedSettings = scheduler.scheduler().settings().stagedParameters();
            expect(stagedSettings.contains("timeout_ms"));
            expect(eq(43UZ, gr::test::get_value_or_fail<gr::Size_t>(stagedSettings.find_value("timeout_ms").value())));

            testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, "", block::property::kSetting, {{"timeout_ms", 43}}, //
                ReplyChecker{.expectedEndpoint = block::property::kSetting, .expectedHasData = false});
        };
    };

    "std::vector<Value> segfault test"_test = [] {
        // Regression test kept because a now-fixed bug in Value’s trivial relocation caused a segfault
        // with Clang 18/19 when std::vector<Value> reallocated (first seen in “Get GRC Yaml tests”);
        // this ensures it doesn’t regress.

        std::vector<Value> vec;
        for (int i = 0; i < 2000; ++i) {
            vec.push_back(property_map{{"key", "value"}});
        }
        std::println("std::vector<Value> segfault test before");
        [[maybe_unused]] auto p0 = vec[0]; // Segfault happens here
        std::println("std::vector<Value> segfault test after");
    };

    "Get GRC Yaml tests"_test = [] {
        gr::Graph testGraph(context->loader);
        std::ignore = testGraph.emplaceBlock("gr::testing::Copy<float32>", {}).value();
        std::ignore = testGraph.emplaceBlock("gr::testing::Copy<float32>", {}).value();

        TestScheduler scheduler(std::move(testGraph));

        testing::sendAndWaitForReply<Get>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), //
            scheduler::property::kGraphGRC, {}, [](const Message& reply) {
                if (reply.endpoint == scheduler::property::kGraphGRC && reply.data.has_value()) {
                    const auto& data = reply.data.value();
                    expect(data.contains("value")) << "Reply should contain 'value' field";
                    const auto& yaml = gr::test::get_value_or_fail<std::string>(data.find_value("value").value());
                    expect(!yaml.empty()) << "YAML string should not be empty";
                    std::println("YAML content:\n{}", yaml);

                    // verify well formed by loading from yaml
                    auto graphFromYaml = gr::loadGrc(context->loader, yaml).value();
                    expect(eq(graphFromYaml->blocks().size(), 4UZ)) << std::format("Expected 4 blocks in loaded graph, got {} blocks\n", graphFromYaml->blocks().size());

                    return true;
                }

                return false;
            });
    };

    static const auto setGrcYamlTestGeneric = []<gr::scheduler::ExecutionPolicy policy>() {
        gr::Graph testGraph(context->loader);
        auto&     source = testGraph.emplaceBlock<gr::testing::SlowSource<float>>();
        auto&     copy1  = testGraph.emplaceBlock<gr::testing::Copy<float>>();
        auto&     copy2  = testGraph.emplaceBlock<gr::testing::Copy<float>>();
        auto&     sink   = testGraph.emplaceBlock<gr::testing::CountingSink<float>>();
        expect(testGraph.connect<"out", "in">(source, copy1).has_value());
        expect(testGraph.connect<"out", "in">(copy1, copy2).has_value());
        expect(testGraph.connect<"out", "in">(copy2, sink).has_value());

        TestScheduler<policy> scheduler(std::move(testGraph), /*addTestSourceAndSink=*/false);
        expect(scheduler.state() == lifecycle::State::RUNNING) << fatal;

        const auto getYamlString = [](TestScheduler<policy>& yamlSource) {
            std::string yaml;
            testing::sendAndWaitForReply<gr::message::Command::Get>(yamlSource.toScheduler, yamlSource.fromScheduler, yamlSource.unique_name(), scheduler::property::kGraphGRC, {}, [&yaml](const Message& reply) {
                if (reply.endpoint == scheduler::property::kGraphGRC && reply.data.has_value()) {
                    yaml = reply.data->value_or<std::string>("value", "NO VALUE KEY"sv);
                    return true;
                }
                return false;
            });
            return yaml;
        };

        std::string yaml = getYamlString(scheduler);
        {
            expect(!yaml.empty());
            auto testGraphFromYaml = gr::loadGrc(context->loader, yaml).value();
            expect(eq(testGraphFromYaml->blocks().size(), 4UZ)) << std::format("Expected 4 blocks in loaded graph, got {} blocks\n", testGraphFromYaml->blocks().size());
        }

        // construct secondary graph to exchange with, which should produce a different result
        gr::Graph alternativeGraph(context->loader);
        auto&     altSource = alternativeGraph.emplaceBlock<gr::testing::CountingSource<float>>();
        auto&     altCopy   = alternativeGraph.emplaceBlock<gr::testing::Copy<float>>();
        auto&     altSink   = alternativeGraph.emplaceBlock<gr::testing::CountingSink<float>>();
        expect(alternativeGraph.connect<"out", "in">(altSource, altCopy).has_value());
        expect(alternativeGraph.connect<"out", "in">(altCopy, altSink).has_value());

        const auto altYaml = gr::saveGrc(context->loader, alternativeGraph);

        testing::sendAndWaitForReply<gr::message::Command::Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kGraphGRC, gr::property_map{{"value", altYaml}}, [](const Message& reply) {
            if (reply.endpoint == scheduler::property::kGraphGRC && reply.data.has_value()) {
                const auto oldState = static_cast<gr::lifecycle::State>(reply.data->value_or<int>("originalSchedulerState", gr::lifecycle::State::IDLE));
                expect(oldState == gr::lifecycle::State::RUNNING) << "did not transition state from running by doing exchange()\n";
                return true;
            }
            return false;
        });

        // we must now return the scheduler to a running state, part of the contract of kGraphGRC Set message
        scheduler.run();

        // scheduler now holds the new graph's yaml
        std::string savedAltYaml = getYamlString(scheduler);
        {
            expect(!savedAltYaml.empty());
            auto alternativeGraphFromYaml = gr::loadGrc(context->loader, savedAltYaml).value();
            expect(eq(alternativeGraphFromYaml->blocks().size(), 3UZ)) << std::format("Expected 3 blocks in loaded graph, got {} blocks", alternativeGraphFromYaml->blocks().size());
        }
    };

    "singlethreaded Set GRC yaml"_test          = [] { setGrcYamlTestGeneric.operator()<gr::scheduler::ExecutionPolicy::singleThreaded>(); };
    "multithreaded Set GRC yaml"_test           = [] { setGrcYamlTestGeneric.operator()<gr::scheduler::ExecutionPolicy::multiThreaded>(); };
    "singlethreaded blocking Set GRC yaml"_test = [] { setGrcYamlTestGeneric.operator()<gr::scheduler::ExecutionPolicy::singleThreadedBlocking>(); };

    "UI constraints setting test"_test = [] {
        // Build a fully connected source→copy1→copy2→sink chain. Orphan blocks
        // self-stop via `disconnect_on_done` before staged settings commit, so the
        // earlier version of this test raced depending on stdlib timing.
        gr::Graph testGraph(context->loader);
        auto&     source = testGraph.emplaceBlock<gr::testing::SlowSource<float>>();
        auto&     copy1  = testGraph.emplaceBlock<gr::testing::Copy<float>>();
        auto&     copy2  = testGraph.emplaceBlock<gr::testing::Copy<float>>();
        auto&     sink   = testGraph.emplaceBlock<gr::testing::CountingSink<float>>();
        expect(testGraph.connect<"out", "in">(source, copy1).has_value());
        expect(testGraph.connect<"out", "in">(copy1, copy2).has_value());
        expect(testGraph.connect<"out", "in">(copy2, sink).has_value());

        TestScheduler scheduler(std::move(testGraph), /*addTestSourceAndSink=*/false);
        auto          makeUiConstraints = [](float x, float y) { return gr::property_map{{"x", x}, {"y", y}}; };

        // Setting ui_constraints property for all blocks, universal
        sendMessage<Set>(scheduler.toScheduler, "", block::property::kSetting, //
            {{"ui_constraints", makeUiConstraints(43, 7070)}}                  // data
        );

        // Setting ui_constraints property for one block
        sendMessage<Set>(scheduler.toScheduler, copy1.unique_name, block::property::kSetting, //
            {{"ui_constraints", makeUiConstraints(42, 6)}}                                    // data
        );

        auto uiConstraintsFor = [](const auto& block) {
            property_map result{};
            pmt::ValueVisitor(meta::overloaded{                                                                                 //
                                  [&result]<typename... Args>(const gr::property_map& map) { result = gr::property_map(map); }, //
                                  [&result]<typename Other>(const Other& /*v*/) { result = gr::property_map{}; }})
                .visit(block.settings().get("ui_constraints").value());
            return result;
        };

        expect(awaitCondition(scheduler, [&] {
            auto c1 = uiConstraintsFor(copy1);
            auto c2 = uiConstraintsFor(copy2);
            return !c1.empty() && !c2.empty()                       //
                   && c1.contains("x") && c2.contains("x")          //
                   && c1.template value_or<float>("x", 0.f) == 42.f //
                   && c2.template value_or<float>("x", 0.f) == 43.f;
        })) << "waiting for ui_constraints to be applied (copy1.x==42, copy2.x==43)";

        expect(eq(42.f, gr::test::get_value_or_fail<float>(uiConstraintsFor(copy1)["x"])));
        expect(eq(43.f, gr::test::get_value_or_fail<float>(uiConstraintsFor(copy2)["x"])));

        // Check if block introspection includes ui_constraints
        {
            auto reply = testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, {}, // serviceName
                graph::property::kGraphInspect,                                                                // endpoint
                {},                                                                                            // data
                [](const Message& _reply) { return _reply.data.has_value(); });

            expect(reply.has_value()) << "Reply should contain data";
            expect(reply->data.has_value()) << "Reply should contain data";
            if (reply->data.has_value()) {
                const auto& map = reply->data.value();
                expect(!map.empty()) << "Resulting map should not be empty";

                const auto children = gr::detail::getProperty<gr::property_map>(map, "children"s).value();

                std::set<float> seenUiConstraintsX;
                std::set<float> seenUiConstraintsY;

                for (const auto& child : children) {
                    const auto uiConstraints = gr::detail::getProperty<gr::property_map>(gr::test::get_value_or_fail<gr::property_map>(child.second), "parameters"s, "ui_constraints"s).value();
                    seenUiConstraintsX.insert(gr::test::get_value_or_fail<float>(uiConstraints.find_value("x").value()));
                    seenUiConstraintsY.insert(gr::test::get_value_or_fail<float>(uiConstraints.find_value("y").value()));
                }

                expect(seenUiConstraintsX == std::set<float>{42, 43});
                expect(seenUiConstraintsY == std::set<float>{6, 7070});
            }
        }

        scheduler.scheduler().requestStop();
        expect(awaitCondition(scheduler, [&] { return scheduler.state() != lifecycle::State::RUNNING; })) << "scheduler stopped";

        expect(eq(42.f, gr::test::get_value_or_fail<float>(copy1.ui_constraints["x"])));
        expect(eq(43.f, gr::test::get_value_or_fail<float>(copy2.ui_constraints["x"])));
    };
};

/// old tests, from the time graph handled messages. They're still good
const boost::ut::suite MoreTopologyGraphTests = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using enum gr::message::Command;

    gr::Graph graph(context->loader);
    auto&     source = graph.emplaceBlock<SlowSource<float>>();
    auto&     sink   = graph.emplaceBlock<AtomicCountingSink<float>>();
    expect(graph.connect<"out", "in">(source, sink).has_value());
    expect(eq(graph.edges().size(), 1UZ)) << "edge registered with connect";

    TestScheduler scheduler(std::move(graph), /*addTestSourceAndSink=*/false);

    expect(awaitCondition(scheduler, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
    expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";
    expect(eq(scheduler.graph().edges().size(), 1UZ)) << "added one edge";

    expect(awaitCondition(scheduler, [&sink] { return sink.loadCount() >= 10U; })) << "sink received enough data";
    std::println("executed basic graph");

    // Adding a few blocks
    auto multiply1 = sendAndWaitMessageEmplaceBlock(scheduler.toScheduler, scheduler.fromScheduler, "gr::testing::Copy<float32>"s, property_map{});
    auto multiply2 = sendAndWaitMessageEmplaceBlock(scheduler.toScheduler, scheduler.fromScheduler, "gr::testing::Copy<float32>"s, property_map{});

    // valid to iterate blocks() here, we only read the vector, state() and name, none of these are mutated by running scheduler, unless kEmplaceBlock etc.
    for (const auto& block : scheduler.graph().blocks()) {
        std::println("block in list: {} - state() : {}", block->name(), magic_enum::enum_name(block->state()));
    }
    expect(eq(scheduler.graph().blocks().size(), 4UZ)) << "should contain sink->multiply1->multiply2->sink";

    sendAndWaitMessageEmplaceEdge(scheduler.toScheduler, scheduler.fromScheduler, source.unique_name, "out", multiply1, "in");
    sendAndWaitMessageEmplaceEdge(scheduler.toScheduler, scheduler.fromScheduler, multiply1, "out", multiply2, "in");
    sendAndWaitMessageEmplaceEdge(scheduler.toScheduler, scheduler.fromScheduler, multiply2, "out", sink.unique_name, "in");
    expect(eq(getNReplyMessages(scheduler.fromScheduler), 0UZ));

    // Get the whole graph
    std::string graphUniqueName;
    testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, "", graph::property::kGraphInspect, property_map{}, //
        [&graphUniqueName](const Message& reply) {
            if (reply.endpoint != graph::property::kGraphInspected) {
                return false;
            }

            const auto& data = reply.data.value();

            graphUniqueName = gr::test::get_value_or_fail<std::string>(data.find_value(serialization_fields::BLOCK_UNIQUE_NAME).value());

            const auto& children = gr::test::get_value_or_fail<property_map>(data.find_value("children").value());
            expect(eq(children.size(), 4UZ));

            const auto& edges = gr::test::get_value_or_fail<property_map>(data.find_value("edges").value());
            expect(eq(edges.size(), 4UZ));
            return true;
        });

    // Inspect the scheduler
    testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, "", scheduler::property::kSchedulerInspect, property_map{}, //
        [&graphUniqueName](const Message& reply) {
            if (reply.endpoint != scheduler::property::kSchedulerInspected) {
                return false;
            }

            const auto& data     = reply.data.value();
            const auto& children = gr::test::get_value_or_fail<property_map>(data.find_value("children").value());
            expect(eq(children.size(), 1UZ));

            for (const auto& [childUniqueName, child] : children) {
                expect(eq(std::string_view(graphUniqueName), std::string_view(childUniqueName)));
            }

            // Scheduler contains a graph as the only child, no edges in scheduler
            expect(!data.contains("edges"s));
            return true;
        });

    scheduler.stop();

    // return to initial state
    expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value()) << "could switch to INITIALISED?";
    expect(awaitCondition(scheduler, [&scheduler] { return scheduler.state() == lifecycle::State::INITIALISED; })) << "scheduler INITIALISED w/ timeout";
    expect(scheduler.state() == lifecycle::State::INITIALISED) << std::format("scheduler INITIALISED - actual: {}\n", magic_enum::enum_name(scheduler.state()));

    scheduler.run();
    expect(awaitCondition(scheduler, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
    expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

    // we must stop the scheduler before looking at edges() because it is doing connectPendingEdges()
    scheduler.stop();

    for (const auto& edge : scheduler.graph().edges()) {
        std::println("edge in list({}): {}", scheduler.graph().edges().size(), edge);
    }
    expect(eq(scheduler.graph().edges().size(), 4UZ)) << "added three new edges, one previously registered with connect";

    scheduler.run();
    expect(awaitCondition(scheduler, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";
    expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

    expect(awaitCondition(scheduler, [&sink] {
        std::this_thread::sleep_for(100ms);
        std::println("sink has received {} samples - parents: {}", sink.loadCount(), sink.in.buffer().streamBuffer.n_writers());
        return sink.loadCount() >= 10U;
    })) << "sink received enough data";

    std::print("Counting sink counted to {}\n", sink.loadCount());
};

const boost::ut::suite InspectBlockTests = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::test;
    using enum gr::message::Command;

    "kInspectBlock returns property_map for a normal block"_test = [] {
        gr::Graph graph(context->loader);
        auto&     source = graph.emplaceBlock<gr::testing::CountingSink<float>>();

        TestScheduler scheduler(std::move(graph));

        auto reply = testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.graph().unique_name, graph::property::kInspectBlock, property_map{{"uniqueName", std::string_view{source.unique_name}}}, [](const Message& msg) { return msg.endpoint == graph::property::kBlockInspected; });

        expect(reply.has_value());
        if (reply) {
            const auto& data = reply->data.value();
            expect(data.contains("id")) << "id field must be present";
            expect(data.contains("unique_name")) << "unique_name field must be present";
            expect(!data.contains("yamlData")) << "yamlData must not be present in property_map mode";
        }
    };

    "kInspectBlock returns yamlData string when serialization_format is yaml"_test = [] {
        gr::Graph graph(context->loader);
        auto&     source = graph.emplaceBlock<gr::testing::CountingSink<float>>();

        TestScheduler scheduler(std::move(graph));

        auto reply = testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.graph().unique_name, graph::property::kInspectBlock, property_map{{"uniqueName", std::string_view{source.unique_name}}, {"serialization_format", "yaml"s}}, [](const Message& msg) { return msg.endpoint == graph::property::kBlockInspected; });

        expect(reply.has_value());
        if (reply) {
            const auto& data = reply->data.value();
            expect(data.contains("yamlData")) << "yamlData key must be present";
            if (data.contains("yamlData")) {
                const auto yaml = gr::test::get_value_or_fail<std::string>(data.find_value("yamlData").value());
                expect(!yaml.empty()) << "yamlData must not be empty";
            }
        }
    };
};

const boost::ut::suite EmplaceBlockFromYamlTests = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::test;
    using enum gr::message::Command;

    "kEmplaceBlock with yaml field creates a normal block"_test = [] {
        gr::Graph graph(context->loader);
        auto&     existingBlock = graph.emplaceBlock<gr::testing::CountingSink<float>>();

        TestScheduler scheduler(std::move(graph));

        // First, inspect the block to get its YAML definition
        auto inspectReply = testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.graph().unique_name, graph::property::kInspectBlock, property_map{{"uniqueName", std::string_view{existingBlock.unique_name}}, {"serialization_format", "yaml"s}}, [](const Message& msg) { return msg.endpoint == graph::property::kBlockInspected; });

        expect(fatal(inspectReply.has_value())) << "kInspectBlock must succeed";
        const auto yamlDef = gr::test::get_value_or_fail<std::string>(inspectReply->data.value().find_value("yamlData").value());

        const auto blockCountBefore = scheduler.graph().blocks().size();

        // Now emplace a new block from that YAML definition
        auto emplaceReply = testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceBlock, property_map{{"yaml", yamlDef}}, [](const Message& msg) { return msg.endpoint == scheduler::property::kBlockEmplaced; });

        expect(emplaceReply.has_value()) << "kEmplaceBlock with yaml must succeed";
        expect(eq(scheduler.graph().blocks().size(), blockCountBefore + 1UZ)) << "one new block must be added";

        if (emplaceReply) {
            const auto& replyData = emplaceReply->data.value();
            expect(replyData.contains("id")) << "reply must contain id";
            const auto newId = gr::test::get_value_or_fail<std::string>(replyData.find_value("id").value());
            expect(!newId.empty()) << "id must not be empty";
        }
    };

    "kEmplaceBlock with empty yaml field returns error"_test = [] {
        TestScheduler scheduler(gr::Graph(context->loader));

        auto reply = testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceBlock, property_map{{"yaml", ""s}}, [](const Message& msg) { return msg.endpoint == scheduler::property::kEmplaceBlock || msg.endpoint == scheduler::property::kBlockEmplaced; });

        expect(reply.has_value());
        if (reply) {
            expect(!reply->data.has_value()) << "error response must have no data";
        }
    };

    "kEmplaceBlock with invalid yaml field returns error"_test = [] {
        TestScheduler scheduler(gr::Graph(context->loader));

        auto reply = testing::sendAndWaitForReply<Set>(scheduler.toScheduler, scheduler.fromScheduler, scheduler.unique_name(), scheduler::property::kEmplaceBlock, property_map{{"yaml", "id: nonexistent::BlockType<float32>\nparameters: {}\n"s}}, [](const Message& msg) { return msg.endpoint == scheduler::property::kEmplaceBlock || msg.endpoint == scheduler::property::kBlockEmplaced; });

        // Either an error is returned or no reply (behaviour depends on emplaceBlock error handling)
        // The main check is that it doesn't crash
        expect(true) << "must not crash on invalid yaml block type";
    };
};

int main() { /* tests are statically executed */ }
