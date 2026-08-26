#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/SubGraph.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>

#include <gnuradio-4.0/testing/DeviceExpectation.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace gr::subgraph_test {

using namespace boost::ut;
using namespace gr;

using Copy   = gr::testing::Copy<float>;
using Source = gr::testing::CountingSource<float>;
using Sink   = gr::testing::CountingSink<float>;

struct Pipeline {
    gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::externalStep> scheduler;
    Sink*                                                               sink   = nullptr;
    gr::SubGraphHandle                                            handle = {};
};

// members are emplaced in `emplacementOrder` but always connected first -> second -> ... , so a reversed order
// exercises the domain's topological sort rather than its emplacement order
std::unique_ptr<Pipeline> makePipeline(std::size_t nMembers, bool reverseEmplacement) {
    gr::Graph                inner;
    std::vector<Copy*>       members(nMembers, nullptr);
    std::vector<std::size_t> emplacementOrder(nMembers);
    std::ranges::generate(emplacementOrder, [n = 0UZ]() mutable { return n++; });
    if (reverseEmplacement) {
        std::ranges::reverse(emplacementOrder);
    }
    for (std::size_t slot : emplacementOrder) {
        members[slot] = std::addressof(inner.emplaceBlock<Copy>());
    }
    for (std::size_t i = 1UZ; i < nMembers; ++i) {
        expect(inner.connect(*members[i - 1UZ], "out", *members[i], "in").has_value());
    }

    auto domain = gr::makeSubGraph(std::move(inner));
    expect(domain.has_value()) << [&] { return domain ? std::string{} : domain.error().message; };

    auto pipeline    = std::make_unique<Pipeline>();
    pipeline->handle = std::move(domain.value());

    gr::Graph graph;
    auto&     src  = graph.emplaceBlock<Source>();
    auto&     snk  = graph.emplaceBlock<Sink>();
    pipeline->sink = std::addressof(snk);

    const auto&       domainRef = graph.addBlock(std::move(pipeline->handle.block));
    const std::string domainName(domainRef->uniqueName());

    expect(graph.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(domainName), pipeline->handle.inputs.at(0), gr::undefined_size, 0, "src->domain").has_value());
    expect(graph.emplaceEdge(std::string_view(domainName), pipeline->handle.outputs.at(0), std::string_view(snk.unique_name), "in", gr::undefined_size, 0, "domain->sink").has_value());

    expect(pipeline->scheduler.exchange(std::move(graph)).has_value());
    expect(pipeline->scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value());
    expect(pipeline->scheduler.changeStateTo(lifecycle::State::RUNNING).has_value());
    return pipeline;
}

std::size_t stepsUntilFirstSample(std::size_t nMembers, bool reverseEmplacement) {
    constexpr std::size_t kCap     = 64UZ;
    auto                  pipeline = makePipeline(nMembers, reverseEmplacement);
    for (std::size_t step = 1UZ; step <= kCap; ++step) {
        std::ignore = pipeline->scheduler.step();
        if (pipeline->sink->count.value > 0U) {
            return step;
        }
    }
    return kCap + 1UZ;
}

const boost::ut::suite<"SubGraph"> _subGraphTests = [] {
    "makeSubGraph exports exactly the ports no interior edge claims, named after their member"_test = [] {
        gr::Graph inner;
        auto&     first  = inner.emplaceBlock<Copy>({{"name", std::string("head")}});
        auto&     second = inner.emplaceBlock<Copy>({{"name", std::string("tail")}});
        expect(inner.connect(first, "out", second, "in").has_value());

        auto domain = gr::makeSubGraph(std::move(inner));
        expect(domain.has_value());
        expect(eq(domain->inputs.size(), 1UZ)) << "only the head's input is unclaimed";
        expect(eq(domain->outputs.size(), 1UZ)) << "only the tail's output is unclaimed";
        // named after the member and its port: adding a member can never rename a port already in use
        expect(eq(domain->inputs.at(0), std::string("head:in"))) << "a boundary port carries the name of the member it belongs to";
        expect(eq(domain->outputs.at(0), std::string("tail:out")));
        expect(domain->block->blockCategory() == block::Category::ScheduledBlockGroup);
        expect(domain->block->asSchedulerModel() != nullptr) << "the parent refuses a group that is not a SchedulerModel";
    };

    // block names are user-provided and default to the TYPE name, so two unnamed members of the same type collide.
    // An ambiguous port lookup shows up as a graph that never finishes, so this must be refused at construction.
    "makeSubGraph refuses two members that would export the same port name"_test = [] {
        gr::Graph inner;
        std::ignore = inner.emplaceBlock<Copy>(); // both unnamed => both named after their type
        std::ignore = inner.emplaceBlock<Copy>();

        auto domain = gr::makeSubGraph(std::move(inner));
        expect(!domain.has_value()) << "two members exporting one name must be an error, not two ports sharing a name";
        if (!domain.has_value()) {
            expect(domain.error().message.contains("must be unique")) << "the error must say what the user has to change";
        }

        gr::Graph named;
        auto&     a = named.emplaceBlock<Copy>({{"name", std::string("alpha")}});
        auto&     b = named.emplaceBlock<Copy>({{"name", std::string("beta")}});
        std::ignore = a;
        std::ignore = b;
        expect(gr::makeSubGraph(std::move(named)).has_value()) << "positive control: the same two members with distinct names are accepted";
    };

    "a domain streams end-to-end inside a parent graph"_test = [] {
        auto pipeline = makePipeline(3UZ, false);
        for (std::size_t i = 0UZ; i < 32UZ; ++i) {
            std::ignore = pipeline->scheduler.step();
        }
        expect(gt(pipeline->sink->count.value, 0U)) << "samples must cross both domain boundaries";

        std::ignore = pipeline->scheduler.changeStateTo(lifecycle::State::REQUESTED_STOP);
        std::ignore = pipeline->scheduler.changeStateTo(lifecycle::State::STOPPED);
    };

    // the payoff question: does an interior edge inside a Domain still get device residency? Graph.hpp marks an edge
    // whose two endpoints share one device domain as DeviceOnly and resolves it to device USM — but that runs in
    // applyEdgeConnection, and a Domain's interior edges are connected by its OWN graph, not the parent's
    "an interior edge between two same-domain device members is marked device-only"_test = [] {
        gr::Graph inner;
        auto&     first  = inner.emplaceBlock<Copy>({{"compute_domain", std::string("gpu:sycl")}});
        auto&     second = inner.emplaceBlock<Copy>({{"compute_domain", std::string("gpu:sycl")}});
        expect(inner.connect(first, "out", second, "in").has_value());

        auto domain = gr::makeSubGraph(std::move(inner));
        expect(domain.has_value());
        BlockModel* domainBlock = domain->block.get();

        gr::Graph                      graph;
        auto&                          src       = graph.emplaceBlock<Source>();
        auto&                          snk       = graph.emplaceBlock<Sink>();
        const std::vector<std::string> inputs    = domain->inputs;
        const std::vector<std::string> outputs   = domain->outputs;
        const auto&                    domainRef = graph.addBlock(std::move(domain->block));
        const std::string              domainName(domainRef->uniqueName());
        expect(graph.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(domainName), inputs.at(0), gr::undefined_size, 0, "src->domain").has_value());
        expect(graph.emplaceEdge(std::string_view(domainName), outputs.at(0), std::string_view(snk.unique_name), "in", gr::undefined_size, 0, "domain->sink").has_value());

        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::externalStep> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value());
        expect(scheduler.changeStateTo(lifecycle::State::RUNNING).has_value()); // triggers startDispatch -> connectPendingEdges

        std::span<Edge> interior = domainBlock->edges();
        expect(eq(interior.size(), 1UZ)) << "the Domain owns exactly the one interior edge";
        if (interior.size() == 1UZ) {
            expect(interior[0]._domain.isDevice()) << "the interior edge must inherit the members' compute_domain";
            expect(interior[0]._domain.access == gr::Access::DeviceOnly) << "an edge inside one device domain never crosses to the host, so it must be device-only";
        }
    };

    // a cycle makes the topological sort degrade to emplacement order, so a member could read the shared buffer
    // before the member that fills it has run -- plausible wrong numbers rather than a failure. The domain must
    // refuse to share buffers at all in that case.
    "members run in dependency order, not emplacement order"_test = [] {
        const std::size_t forward = stepsUntilFirstSample(3UZ, false);
        const std::size_t reverse = stepsUntilFirstSample(3UZ, true);
        expect(le(forward, 8UZ)) << "a sorted chain should traverse in a couple of steps";
        expect(eq(reverse, forward)) << "reversing emplacement must not cost a step; the domain sorts topologically";
    };

    // registerSyclRuntime() mutates the process-global ComputeRegistry, so this test runs last: everything above it
    // must keep observing the pre-registration behaviour (a gpu:sycl domain that resolves to no real USM resource).
    "an interior edge between two same-domain device members lives in memory the device can actually dereference"_test = [] {
        if (!gr::device::registerSyclRuntime()) {
            expect(!gr::testing::deviceDomainRequired("gpu:sycl")) << "GR4_REQUIRE_DEVICE names gpu:sycl but no SYCL backend was compiled in";
            boost::ut::log << "skipped: this build has no SYCL backend compiled in";
            return;
        }
        auto* ctx = gr::device::DeviceContextRegistry::instance().tryResolve("gpu:sycl");
        if (ctx == nullptr) {
            expect(!gr::testing::deviceDomainRequired("gpu:sycl")) << "GR4_REQUIRE_DEVICE names gpu:sycl, so this lane must exercise it rather than skip";
            boost::ut::log << "skipped: no gpu:sycl device is registered on this machine";
            return;
        }

        gr::Graph inner;
        auto&     first  = inner.emplaceBlock<Copy>({{"compute_domain", std::string("gpu:sycl")}});
        auto&     second = inner.emplaceBlock<Copy>({{"compute_domain", std::string("gpu:sycl")}});
        expect(inner.connect(first, "out", second, "in").has_value()) << "the two device members must connect before the domain can be built";

        auto domain = gr::makeSubGraph(std::move(inner));
        expect(domain.has_value()) << "makeSubGraph must succeed for two device-domain members joined by one interior edge";
        BlockModel* domainBlock = domain->block.get();

        gr::Graph                      graph;
        auto&                          src       = graph.emplaceBlock<Source>();
        auto&                          snk       = graph.emplaceBlock<Sink>();
        const std::vector<std::string> inputs    = domain->inputs;
        const std::vector<std::string> outputs   = domain->outputs;
        const auto&                    domainRef = graph.addBlock(std::move(domain->block));
        const std::string              domainName(domainRef->uniqueName());
        expect(graph.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(domainName), inputs.at(0), gr::undefined_size, 0, "src->domain").has_value()) << "the source must reach the domain's boundary input";
        expect(graph.emplaceEdge(std::string_view(domainName), outputs.at(0), std::string_view(snk.unique_name), "in", gr::undefined_size, 0, "domain->sink").has_value()) << "the domain's boundary output must reach the sink";

        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::externalStep> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value()) << "the parent graph must install into the scheduler";
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value()) << "the scheduler must reach INITIALISED before RUNNING";
        expect(scheduler.changeStateTo(lifecycle::State::RUNNING).has_value()) << "RUNNING triggers startDispatch -> connectPendingEdges, which resolves the interior edge's data resource";

        std::span<Edge> interior = domainBlock->edges();
        expect(eq(interior.size(), 1UZ)) << "the Domain owns exactly the one interior edge";
        if (interior.size() == 1UZ) {
            expect(interior[0]._domain.access == gr::Access::DeviceOnly) << "an edge interior to one device domain must be marked device-only before its resource is even resolved";

            std::pmr::memory_resource* resource = interior[0]._dataResource;
            expect(resource != nullptr) << "the interior edge must have a resolved data resource once the domain is running";
            if (resource != nullptr) {
                // a double-mapping resource aliases its second half at a whole multiple of its granule, so an
                // allocation must be sized in granules; 0 means the resource has no such constraint and any size will do
                const std::size_t granularity = gr::allocationGranularity(resource);
                const std::size_t nBytes      = granularity == 0UZ ? sizeof(float) : granularity;
                void*             ptr         = resource->allocate(nBytes, alignof(std::max_align_t));
                expect(ctx->isDeviceAccessible(ptr)) << "an interior edge inside one device domain must allocate memory the device can actually dereference, not merely memory tagged device-only";
                resource->deallocate(ptr, nBytes, alignof(std::max_align_t));
            }
        }
    };

    // two exported-port hops through a private inner graph is the likeliest place for a tag to be lost. The key is
    // `gr:`-prefixed because that is the set the framework auto-forwards.
    "a gr: tag survives both domain boundaries at its original sample index"_test = [] {
        using gr::testing::ProcessFunction;
        constexpr gr::Size_t  kSamples  = 1024U;
        constexpr std::size_t kTagIndex = 3UZ;

        gr::Graph inner;
        auto&     first  = inner.emplaceBlock<Copy>();
        auto&     second = inner.emplaceBlock<Copy>();
        expect(inner.connect(first, "out", second, "in").has_value());
        auto domain = gr::makeSubGraph(std::move(inner));
        expect(domain.has_value());

        gr::Graph flow;
        auto&     src = flow.emplaceBlock<gr::testing::TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kSamples}, {"mark_tag", false}});
        auto&     snk = flow.emplaceBlock<gr::testing::TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", kSamples}, {"log_tags", true}, {"log_samples", true}});

        gr::property_map payload;
        gr::tag::put(payload, "gr:trigger_name", gr::pmt::Value(std::string("domain-boundary")));
        src._tags.push_back(gr::testing::OwningTag{kTagIndex, payload});

        const std::vector<std::string> inputs    = domain->inputs;
        const std::vector<std::string> outputs   = domain->outputs;
        const auto&                    domainRef = flow.addBlock(std::move(domain->block));
        const std::string              domainName(domainRef->uniqueName());
        expect(flow.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(domainName), inputs.at(0), gr::undefined_size, 0, "src->domain").has_value());
        expect(flow.emplaceEdge(std::string_view(domainName), outputs.at(0), std::string_view(snk.unique_name), "in", gr::undefined_size, 0, "domain->sink").has_value());

        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::externalStep> scheduler;
        expect(scheduler.exchange(std::move(flow)).has_value());
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value());
        expect(scheduler.changeStateTo(lifecycle::State::RUNNING).has_value());
        for (std::size_t i = 0UZ; i < 256UZ && snk._samples.size() < static_cast<std::size_t>(kSamples); ++i) {
            std::ignore = scheduler.step();
        }

        expect(ge(snk._samples.size(), static_cast<std::size_t>(kSamples))) << "the samples themselves must arrive before the tag can be judged";
        expect(eq(snk._tags.size(), 1UZ)) << "exactly the one published tag must cross both domain boundaries";
        if (snk._tags.size() == 1UZ) {
            expect(eq(snk._tags[0].index, kTagIndex)) << "a tag that arrives at the wrong sample index is as broken as one that is lost";
            expect(snk._tags[0].map.contains("gr:trigger_name")) << "the forwarded key must survive the trip, not just the tag's position";
        }

        std::ignore = scheduler.changeStateTo(lifecycle::State::REQUESTED_STOP);
        std::ignore = scheduler.changeStateTo(lifecycle::State::STOPPED);
    };

    // makeSubGraph used to set disconnect_on_done=false on every member; settings().set() through a BlockModel
    // never reached the field, so it was a no-op for its whole life. This pins the behaviour that actually holds.
    "makeSubGraph leaves its members' settings alone"_test = [] {
        gr::Graph inner;
        auto&     head   = inner.emplaceBlock<Copy>({{"name", std::string("head")}});
        auto&     middle = inner.emplaceBlock<Copy>({{"name", std::string("middle")}});
        auto&     tail   = inner.emplaceBlock<Copy>({{"name", std::string("tail")}});
        expect(inner.connect(head, "out", middle, "in").has_value());
        expect(inner.connect(middle, "out", tail, "in").has_value());
        expect(head.disconnect_on_done && middle.disconnect_on_done && tail.disconnect_on_done) << "precondition: all three start at the default";

        auto domain = gr::makeSubGraph(std::move(inner));
        expect(domain.has_value());
        static_cast<gr::SubGraphWrapper*>(domain->block.get())->start();

        expect(head.disconnect_on_done) << "a boundary member's settings are the caller's, not the helper's";
        expect(middle.disconnect_on_done) << "and so are an interior member's";
        expect(tail.disconnect_on_done);
    };

    "makeSubGraph keeps a port private when asked, and exports the rest"_test = [] {
        gr::Graph inner;
        auto&     head = inner.emplaceBlock<Copy>({{"name", std::string("head")}});
        auto&     tail = inner.emplaceBlock<Copy>({{"name", std::string("tail")}});
        expect(inner.connect(head, "out", tail, "in").has_value());

        auto domain = gr::makeSubGraph(std::move(inner), {"tail:out"});
        expect(domain.has_value());
        expect(eq(domain->inputs.size(), 1UZ)) << "the head's input is still exported";
        expect(eq(domain->inputs.at(0), std::string("head:in")));
        expect(eq(domain->outputs.size(), 0UZ)) << "the port named in doNotExport must stay private";
    };

    // one device domain plus host members is a legitimate mix; two DEVICE domains are not, because only one could
    // have its buffers bound and the other would fall back to the host path with nothing saying so
    "makeSubGraph refuses members spread across two device domains"_test = [] {
        gr::Graph mixed;
        auto&     cuda = mixed.emplaceBlock<Copy>({{"name", std::string("a")}, {"compute_domain", std::string("gpu:cuda")}});
        auto&     sycl = mixed.emplaceBlock<Copy>({{"name", std::string("b")}, {"compute_domain", std::string("gpu:sycl")}});
        expect(mixed.connect(cuda, "out", sycl, "in").has_value());

        auto refused = gr::makeSubGraph(std::move(mixed));
        expect(!refused.has_value()) << "two device domains in one group must be an error, not a silent host fallback for one of them";
        if (!refused.has_value()) {
            expect(refused.error().message.contains("at most one device")) << "the error must say what the constraint is";
        }

        // positive control: the SAME shape with one device domain and a host member is accepted
        gr::Graph allowed;
        auto&     host   = allowed.emplaceBlock<Copy>({{"name", std::string("a")}});
        auto&     device = allowed.emplaceBlock<Copy>({{"name", std::string("b")}, {"compute_domain", std::string("gpu:cuda")}});
        expect(allowed.connect(host, "out", device, "in").has_value());
        expect(gr::makeSubGraph(std::move(allowed)).has_value()) << "a host member alongside one device domain is a legitimate mix";
    };
};

} // namespace gr_domain_test

int main() { /* tests are statically executed */ }
