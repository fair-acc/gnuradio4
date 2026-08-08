#include <boost/ut.hpp>

#include <thread>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/SchedulerModel.hpp>

#include <gnuradio-4.0/testing/NullSources.hpp>

namespace gr::dispatch_group_test {

using namespace boost::ut;
using namespace gr;

/**
 * @brief CPU stand-in for a device syncGroup: a ScheduledBlockGroup that runs its members from its own work().
 *
 * Pins down two traps the framework sets for a block group:
 * - `Block<Derived>::work()` returns immediately for any category other than `NormalBlock`, so a group must declare
 *   its own `work()` and rely on `BlockWrapper::work()` forwarding non-virtually.
 * - `BlockWrapper::dynamicPortsLoader()` registers static ports only for a `NormalBlock`, so boundary ports come
 *   from `GraphWrapper::exportPort`.
 */
struct PassthroughDispatchScheduler : gr::Block<PassthroughDispatchScheduler> {
    using Description = Doc<"runs its member blocks synchronously from its own work(), with no scheduler thread">;

    GR_MAKE_REFLECTABLE(PassthroughDispatchScheduler);

    constexpr static block::Category blockCategory = block::Category::ScheduledBlockGroup;

    meta::indirect<gr::Graph> _graph{};
    std::size_t               _dispatchCalls = 0UZ;
    std::size_t               _memberCalls   = 0UZ;
    std::thread::id           _dispatchThread{};
    bool                      _quiescent = false;

    // GraphWrapper forwards findPortInBlock/blocks/edges to the wrapped type, so a group must expose them
    [[nodiscard]] const gr::Graph& graph() const noexcept { return *_graph; }
    [[nodiscard]] gr::Graph&       graph() noexcept { return *_graph; }

    [[nodiscard]] std::span<std::shared_ptr<BlockModel>>       blocks() noexcept { return _graph->blocks(); }
    [[nodiscard]] std::span<const std::shared_ptr<BlockModel>> blocks() const noexcept { return _graph->blocks(); }
    [[nodiscard]] std::span<Edge>                              edges() noexcept { return _graph->edges(); }
    [[nodiscard]] std::span<const Edge>                        edges() const noexcept { return _graph->edges(); }

    void setGraph(gr::Graph&& newGraph) { _graph = meta::indirect<gr::Graph>(std::move(newGraph)); }

    void startDispatch() {
        std::ignore = _graph->connectPendingEdges(); // interior edges only; the boundary ones belong to the parent
        for (auto& member : _graph->blocks()) {
            std::ignore = member->changeStateTo(lifecycle::State::INITIALISED);
            std::ignore = member->changeStateTo(lifecycle::State::RUNNING);
        }
    }

    void stopDispatch() {
        for (auto& member : _graph->blocks()) {
            std::ignore = member->changeStateTo(lifecycle::State::REQUESTED_STOP);
            std::ignore = member->changeStateTo(lifecycle::State::STOPPED);
        }
    }

    // no wait loop: work() only ever runs on the thread that calls it, so quiescence cannot race with a member
    void requestWorkQuiescence() { _quiescent = true; }
    void releaseWorkQuiescence() { _quiescent = false; }

    [[nodiscard]] work::Result work(std::size_t requestedWork) noexcept {
        if (_quiescent) {
            return {requestedWork, 0UZ, work::Status::OK};
        }

        ++_dispatchCalls;
        _dispatchThread = std::this_thread::get_id();

        std::size_t performed = 0UZ;
        for (auto& member : _graph->blocks()) {
            performed += member->work(requestedWork).performed_work;
            ++_memberCalls;
        }
        return {requestedWork, performed, work::Status::OK};
    }
};

// SchedulerWrapper without the std::thread: start() primes the members and returns, leaving the parent's work
// loop as the only thing that drives the syncGroup.
template<typename TDispatch>
class DispatchWrapper : public GraphWrapper<TDispatch, gr::Graph>, public SchedulerModel {
public:
    explicit DispatchWrapper(gr::property_map props = {}) : GraphWrapper<TDispatch, gr::Graph>(std::move(props)) {}

    void            setGraph(gr::Graph&& graph) final { this->blockRef().setGraph(std::move(graph)); }
    BlockModel*     asBlockModel() final { return static_cast<BlockModel*>(this); }
    SchedulerModel* asSchedulerModel() noexcept override { return this; }

    void start() override { this->blockRef().startDispatch(); }
    void stop() override { this->blockRef().stopDispatch(); }

    // no workers of its own: members run on whoever calls work(), so quiescence is a flag and there is nothing to
    // recurse into, nothing to wait for, and no adopted blocks to hand back
    void requestWorkQuiescenceAll() override { this->blockRef().requestWorkQuiescence(); }
    void releaseWorkQuiescenceAll() override { this->blockRef().releaseWorkQuiescence(); }
    void blockUntilWorking() override {}
    void removeBlocks(std::span<const std::shared_ptr<BlockModel>>) final {}
};

using Dispatch = DispatchWrapper<PassthroughDispatchScheduler>;

struct SyncGroup {
    std::shared_ptr<gr::BlockModel> owned;
    Dispatch*                       wrapper = nullptr;
    std::string                     name;
};

SyncGroup makeSyncGroup() {
    gr::Graph inner;
    auto&     first  = inner.emplaceBlock<gr::testing::Copy<float>>();
    auto&     second = inner.emplaceBlock<gr::testing::Copy<float>>();
    // both boundary peers are attached by the parent, so a member must not stop for want of a neighbour
    first.disconnect_on_done  = false;
    second.disconnect_on_done = false;
    expect(inner.connect(first, "out", second, "in").has_value());

    const std::string firstName(first.unique_name);
    const std::string secondName(second.unique_name);

    SyncGroup syncGroup;
    syncGroup.owned   = std::static_pointer_cast<gr::BlockModel>(std::make_shared<Dispatch>());
    syncGroup.wrapper = static_cast<Dispatch*>(syncGroup.owned.get());
    syncGroup.wrapper->setGraph(std::move(inner));

    expect(syncGroup.wrapper->exportPort(true, firstName, gr::PortDirection::INPUT, "in", "inExp").has_value());
    expect(syncGroup.wrapper->exportPort(true, secondName, gr::PortDirection::OUTPUT, "out", "outExp").has_value());
    return syncGroup;
}

const boost::ut::suite<"DispatchGroup"> _dispatchGroupTests = [] {
    using enum gr::lifecycle::State;

    "a ScheduledBlockGroup with its own work() is driven synchronously by the parent's step()"_test = [] {
        SyncGroup syncGroup = makeSyncGroup();

        gr::Graph graph;
        auto&     src  = graph.emplaceBlock<gr::testing::CountingSource<float>>();
        auto&     sink = graph.emplaceBlock<gr::testing::CountingSink<float>>();

        const auto&       groupRef = graph.addBlock(std::move(syncGroup.owned));
        const std::string groupName(groupRef->uniqueName());

        expect(graph.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(groupName), "inExp", gr::undefined_size, 0, "src->syncGroup").has_value());
        expect(graph.emplaceEdge(std::string_view(groupName), "outExp", std::string_view(sink.unique_name), "in", gr::undefined_size, 0, "syncGroup->sink").has_value());

        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::externalStep> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.changeStateTo(INITIALISED).has_value());
        expect(scheduler.changeStateTo(RUNNING).has_value());

        for (std::size_t i = 0UZ; i < 64UZ; ++i) {
            std::ignore = scheduler.step();
        }

        const PassthroughDispatchScheduler& dispatch = syncGroup.wrapper->blockRef();

        expect(gt(dispatch._dispatchCalls, 0UZ)) << "our work() must shadow Block::work(), which is inert for a ScheduledBlockGroup";
        expect(gt(dispatch._memberCalls, 0UZ)) << "the syncGroup must have driven its members";
        expect(gt(sink.count.value, 0U)) << "samples must cross both syncGroup boundaries";
        expect(dispatch._dispatchThread == std::this_thread::get_id()) << "the syncGroup must run on the caller's thread, not one of its own";

        std::ignore = scheduler.changeStateTo(REQUESTED_STOP);
        std::ignore = scheduler.changeStateTo(STOPPED);
    };

    "the sink only advances while the syncGroup dispatches"_test = [] {
        SyncGroup syncGroup = makeSyncGroup();

        gr::Graph graph;
        auto&     src  = graph.emplaceBlock<gr::testing::CountingSource<float>>();
        auto&     sink = graph.emplaceBlock<gr::testing::CountingSink<float>>();

        const auto&       groupRef = graph.addBlock(std::move(syncGroup.owned));
        const std::string groupName(groupRef->uniqueName());

        expect(graph.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(groupName), "inExp", gr::undefined_size, 0, "src->syncGroup").has_value());
        expect(graph.emplaceEdge(std::string_view(groupName), "outExp", std::string_view(sink.unique_name), "in", gr::undefined_size, 0, "syncGroup->sink").has_value());

        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::externalStep> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.changeStateTo(INITIALISED).has_value());
        expect(scheduler.changeStateTo(RUNNING).has_value());

        for (std::size_t i = 0UZ; i < 64UZ; ++i) {
            std::ignore = scheduler.step();
        }
        const gr::Size_t whileDispatching = sink.count.value;
        expect(gt(whileDispatching, 0U));

        // silence the syncGroup and let the backlog drain, then prove nothing more arrives: the data path runs
        // through our work(), it does not merely coexist with it
        syncGroup.wrapper->requestWorkQuiescenceAll();
        for (std::size_t i = 0UZ; i < 64UZ; ++i) {
            std::ignore = scheduler.step();
        }
        const std::size_t callsAfterDrain = syncGroup.wrapper->blockRef()._dispatchCalls;
        const gr::Size_t  countAfterDrain = sink.count.value;

        for (std::size_t i = 0UZ; i < 64UZ; ++i) {
            std::ignore = scheduler.step();
        }
        expect(eq(sink.count.value, countAfterDrain)) << "a quiescent syncGroup must starve its downstream";
        expect(eq(syncGroup.wrapper->blockRef()._dispatchCalls, callsAfterDrain)) << "a quiescent syncGroup must not dispatch";

        syncGroup.wrapper->releaseWorkQuiescenceAll();
        for (std::size_t i = 0UZ; i < 64UZ; ++i) {
            std::ignore = scheduler.step();
        }
        expect(gt(sink.count.value, countAfterDrain)) << "releasing quiescence must resume the flow";

        std::ignore = scheduler.changeStateTo(REQUESTED_STOP);
        std::ignore = scheduler.changeStateTo(STOPPED);
    };
};

} // namespace gr::dispatch_group_test

int main() { /* tests are statically executed */ }
