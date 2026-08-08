#include <boost/ut.hpp>

#include <format>
#include <memory_resource>
#include <print>

#include <gnuradio-4.0/DeviceGroup.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/BridgedDeviceGroup.hpp>
#include <gnuradio-4.0/basic/TransferBlocks.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>

#include "device_test_helpers.hpp"
#include <gnuradio-4.0/testing/DeviceExpectation.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

/*
 * The vertical stack a device DeviceGroup is meant to carry:
 *
 *   source ─▶ DeviceGroup[ HostToDevice ─▶ member ─▶ member ─▶ DeviceToHost ] ─▶ sink
 *
 * The transfers live INSIDE the group on purpose. Membership is what declares the boundary, so the crossing points
 * belong to the group that owns it -- and this is the shape an automatic insertion helper has to produce, which
 * makes this file its specification as much as its test.
 */
namespace gr::subgraph_vertical_test {

using namespace boost::ut;
using namespace gr;

// deliberately test-local and unregistered: gr::testing::Copy is also compiled into the block library, and the
// question this file first had to answer was whether that second definition is what breaks the JIT
struct Copy : gr::Block<Copy> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;
    GR_MAKE_REFLECTABLE(Copy, in, out);
    [[nodiscard]] constexpr float processOne(float v) const noexcept { return v; }
};
using Src = gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>;
using Snk = gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_ONE>;

constexpr gr::Size_t kSamples = 4096U;

std::vector<float> runGroup(gr::DeviceGroupHandle group, gr::Size_t nSamples = kSamples);

/// builds the group above and runs it to completion, returning what reached the sink
std::vector<float> runVertical(std::string_view memberDomain, std::size_t nMembers, gr::Size_t nSamples = kSamples) {
    gr::Graph inner;
    auto&     h2d = inner.emplaceBlock<gr::basic::HostToDevice<float>>();
    auto&     d2h = inner.emplaceBlock<gr::basic::DeviceToHost<float>>();

    std::vector<Copy*> members;
    for (std::size_t i = 0UZ; i < nMembers; ++i) {
        members.push_back(std::addressof(inner.emplaceBlock<Copy>({{"compute_domain", std::string(memberDomain)}})));
    }

    expect(inner.connect(h2d, "out", *members.front(), "in").has_value());
    for (std::size_t i = 1UZ; i < nMembers; ++i) {
        expect(inner.connect(*members[i - 1UZ], "out", *members[i], "in").has_value());
    }
    expect(inner.connect(*members.back(), "out", d2h, "in").has_value());

    auto group = gr::makeDeviceGroup(std::move(inner));
    expect(group.has_value()) << [&] { return group ? std::string{} : group.error().message; };
    if (!group) {
        return {};
    }
    return runGroup(std::move(group.value()), nSamples);
}

/// drives an already-built group from a source to a sink and returns what arrived
std::vector<float> runGroup(gr::DeviceGroupHandle group, gr::Size_t nSamples) {
    gr::Graph outer;
    auto&     src = outer.emplaceBlock<Src>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     snk = outer.emplaceBlock<Snk>({{"n_samples_expected", nSamples}, {"log_samples", true}});

    const auto&       added = outer.addBlock(std::move(group.block));
    const std::string name(added->uniqueName());
    expect(outer.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(name), group.inputs.at(0), gr::undefined_size, 0, "src->group").has_value());
    expect(outer.emplaceEdge(std::string_view(name), group.outputs.at(0), std::string_view(snk.unique_name), "in", gr::undefined_size, 0, "group->sink").has_value());

    gr::scheduler::Simple<> scheduler;
    expect(scheduler.exchange(std::move(outer)).has_value());
    expect(scheduler.runAndWait().has_value()) << "the vertical stack must run to completion";

    return {snk._samples.begin(), snk._samples.end()};
}

} // namespace gr::subgraph_vertical_test

int main() {
    using namespace boost::ut;
    using namespace gr::subgraph_vertical_test;

    "a host group carries samples through both transfer blocks unchanged"_test = [] {
        // host control first: the transfers are no-ops here, so anything that breaks below is the device path and
        // not the topology
        const std::vector<float> samples = runVertical("host", 2UZ);
        expect(eq(samples.size(), static_cast<std::size_t>(kSamples))) << "every sample must reach the sink";
        expect(samples.size() < 8UZ || eq(samples[7], 7.f)) << "a Copy chain must not alter the data";
    };

    "the same group on a device produces the same samples"_test = [] {
        std::ignore                                  = gr::device::registerSyclRuntime();
        std::ignore                                  = gr::test::requireHostSycl();
        const std::optional<std::string_view> domain = gr::test::firstServedSyclDomain();
        if (!domain) {
            expect(!gr::testing::deviceDomainRequired("host:sycl")) << "GR4_REQUIRE_DEVICE names a SYCL domain, so this lane must exercise it rather than skip";
            boost::ut::log << "skipped: no SYCL domain is served here";
            return;
        }

        const std::vector<float> onHost   = runVertical("host", 2UZ);
        const std::vector<float> onDevice = runVertical(*domain, 2UZ);

        expect(eq(onDevice.size(), onHost.size())) << "the device leg must deliver the same number of samples";
        expect(std::ranges::equal(onHost, onDevice)) << "a device round trip must be bit-identical for a Copy chain";
    };

    "a device-to-device chain does not add a barrier per member"_test = [] {
        std::ignore                                  = gr::device::registerSyclRuntime();
        const std::optional<std::string_view> served = gr::test::firstServedSyclDomain();
        if (!served) {
            return;
        }
        gr::device::DeviceContext* const ctx = gr::device::DeviceContextRegistry::instance().tryResolve(*served);
        if (ctx == nullptr) {
            return;
        }

        // the claim rests on interior edges living in memory the host cannot dereference. A SYCL CPU device has no
        // separate address space, so its USM device allocation reads back as ordinary host memory, no dispatch may
        // defer its completion, and every hop synchronises -- correctly, and with nothing left to measure here.
        if (std::pmr::memory_resource* deviceOnly = ctx->resource(gr::Access::DeviceOnly); deviceOnly != nullptr) {
            void* const probe           = deviceOnly->allocate(alignof(std::max_align_t), alignof(std::max_align_t));
            const bool  opaqueToTheHost = ctx->isDeviceOnly(probe);
            deviceOnly->deallocate(probe, alignof(std::max_align_t), alignof(std::max_align_t));
            if (!opaqueToTheHost) {
                boost::ut::log << std::format("skipped: '{}' cannot hold memory the host may not dereference", *served);
                return;
            }
        }

        // interior Copy->Copy edges are device-only, so those dispatches return once the kernel is enqueued; only
        // the two boundary hops still synchronise. The count must therefore be flat in the number of members --
        // which is the whole claim of the cascade, and the one thing a throughput number cannot demonstrate.
        const auto barriersFor = [&](std::size_t nMembers, gr::Size_t nSamples) {
            const std::uint64_t      before  = ctx->syncCount();
            const std::vector<float> samples = runVertical(*served, nMembers, nSamples);
            expect(eq(samples.size(), static_cast<std::size_t>(nSamples))) << "the chain must still deliver every sample";
            return ctx->syncCount() - before;
        };

        // A difference of differences. A run costs a fixed teardown (one drain per member) plus a per-work() cost
        // times the number of work() calls, and only the second is what a barrier between hops would inflate. Taking
        // the SAME chain at two stream lengths cancels the teardown; what remains is the per-work() cost, and the
        // claim is that it does not grow with the chain, because an interior hop is device-only on both sides and
        // its kernel is never awaited.
        const auto perWorkBarriers = [&](std::size_t nMembers) { return barriersFor(nMembers, 4U * kSamples) - barriersFor(nMembers, kSamples); };

        const std::uint64_t twoMembers  = perWorkBarriers(2UZ);
        const std::uint64_t fiveMembers = perWorkBarriers(5UZ);
        std::println("  device barriers per extra work(): 2 members = {}, 5 members = {}", twoMembers, fiveMembers);

        expect(gt(twoMembers, 0UZ)) << "a device run must synchronise at its host boundary, or this measures nothing";
        expect(eq(fiveMembers, twoMembers)) << std::format("three added device-resident members must add no per-work() barrier at all, got {} -> {}", twoMembers, fiveMembers);
    };

    "a longer device chain still round-trips"_test = [] {
        std::ignore                                  = gr::device::registerSyclRuntime();
        const std::optional<std::string_view> domain = gr::test::firstServedSyclDomain();
        if (!domain) {
            boost::ut::log << "skipped: no SYCL domain is served here";
            return;
        }
        // four members, so the interior carries three edges the host never sees
        const std::vector<float> samples = runVertical(*domain, 4UZ);
        expect(eq(samples.size(), static_cast<std::size_t>(kSamples)));
        expect(samples.size() < 8UZ || eq(samples[7], 7.f)) << "chain length must not change the result";
    };

    "a group whose members span two device domains is refused, not silently split"_test = [] {
        gr::Graph inner;
        auto&     first  = inner.emplaceBlock<Copy>({{"compute_domain", std::string("gpu:sycl")}});
        auto&     second = inner.emplaceBlock<Copy>({{"compute_domain", std::string("gpu:cuda")}});
        expect(inner.connect(first, "out", second, "in").has_value());

        const auto group = gr::makeDeviceGroup(std::move(inner));
        expect(!group.has_value()) << "two device domains in one group must be an error at construction, not a silent fallback at run time";
    };

    "the transfer blocks are ordinary members: the group exports their outer ports, not the members'"_test = [] {
        gr::Graph inner;
        auto&     h2d    = inner.emplaceBlock<gr::basic::HostToDevice<float>>();
        auto&     member = inner.emplaceBlock<Copy>();
        auto&     d2h    = inner.emplaceBlock<gr::basic::DeviceToHost<float>>();
        expect(inner.connect(h2d, "out", member, "in").has_value());
        expect(inner.connect(member, "out", d2h, "in").has_value());

        const auto group = gr::makeDeviceGroup(std::move(inner));
        expect(group.has_value()) << [&] { return group ? std::string{} : group.error().message; };
        if (!group) {
            return;
        }
        expect(eq(group->inputs.size(), 1UZ)) << "only HostToDevice::in is unclaimed on the input side";
        expect(eq(group->outputs.size(), 1UZ)) << "only DeviceToHost::out is unclaimed on the output side";
        expect(group->inputs.at(0).contains("HostToDevice") && group->inputs.at(0).ends_with(":in")) << "the exported name must identify the member and the port, got: " << group->inputs.at(0);
        expect(group->outputs.at(0).contains("DeviceToHost") && group->outputs.at(0).ends_with(":out")) << "the exported name must identify the member and the port, got: " << group->outputs.at(0);
    };

    "makeBridgedDeviceGroup builds by itself what the manual wiring above builds by hand"_test = [] {
        gr::Graph inner;
        auto&     first  = inner.emplaceBlock<Copy>();
        auto&     second = inner.emplaceBlock<Copy>();
        expect(inner.connect(first, "out", second, "in").has_value());

        const auto group = gr::basic::makeBridgedDeviceGroup<float>(std::move(inner));
        expect(group.has_value()) << [&] { return group ? std::string{} : group.error().message; };
        if (!group) {
            return;
        }
        // the members' own ports are no longer the boundary: a transfer block sits in front of each
        expect(eq(group->inputs.size(), 1UZ));
        expect(eq(group->outputs.size(), 1UZ));
        expect(group->inputs.at(0).contains("h2d_")) << "the group must export the transfer's port, not the member's, got: " << group->inputs.at(0);
        expect(group->outputs.at(0).contains("d2h_")) << "the group must export the transfer's port, not the member's, got: " << group->outputs.at(0);
    };

    "makeBridgedDeviceGroup gives a member with no domain of its own the group's"_test = [] {
        std::ignore                                  = gr::device::registerSyclRuntime();
        const std::optional<std::string_view> served = gr::test::firstServedSyclDomain();
        if (!served) {
            return; // no SYCL device on this machine
        }

        gr::Graph inner;
        inner.compute_domain = std::string(*served); // the group's domain, set before any member is emplaced
        auto& first          = inner.emplaceBlock<Copy>();
        auto& second         = inner.emplaceBlock<Copy>({{"compute_domain", std::string("host")}}); // chosen: must survive
        expect(inner.connect(first, "out", second, "in").has_value());

        auto group = gr::basic::makeBridgedDeviceGroup<float>(std::move(inner));
        expect(group.has_value()) << [&] { return group ? std::string{} : group.error().message; };
        if (!group) {
            return;
        }

        // `init()` applied it, so this reads the active value; the staged branch remains for a domain set later
        const auto domainOf = [](const gr::BlockModel& block) -> std::string {
            const auto& staged = block.settings().stagedParameters();
            if (auto it = staged.find(std::string_view{"compute_domain"}); it != staged.end()) {
                return std::string((*it).second.value_or(std::string_view{}));
            }
            if (auto active = block.settings().get("compute_domain")) {
                return std::string(active->value_or(std::string_view{}));
            }
            return {};
        };
        std::size_t onTheDevice = 0UZ;
        std::size_t onTheHost   = 0UZ;
        for (const std::shared_ptr<gr::BlockModel>& member : static_cast<gr::DeviceGroupWrapper*>(group->block.get())->blockRef().graph().blocks()) {
            const std::string domain = domainOf(*member);
            onTheDevice += domain == *served ? 1UZ : 0UZ;
            onTheHost += domain == "host" ? 1UZ : 0UZ;
        }
        expect(eq(onTheDevice, 3UZ)) << "the domainless member and both transfers must carry the group's domain";
        expect(eq(onTheHost, 1UZ)) << "a member that chose 'host' keeps it -- a host member inside a device group is legal";

        const std::size_t refusals = gr::test::deviceRefusalsDuring([&] { std::ignore = runGroup(std::move(group.value())); });
        expect(eq(refusals, 0UZ)) << "the hoisted member must reach a kernel, not be refused";
    };

    "two device groups chain device-to-device, with no host hop between them"_test = [] {
        std::ignore                                  = gr::device::registerSyclRuntime();
        const std::optional<std::string_view> served = gr::test::firstServedSyclDomain();
        if (!served) {
            return;
        }

        const auto deviceGroup = [&]() {
            gr::Graph inner;
            inner.compute_domain = std::string(*served);
            auto& only           = inner.emplaceBlock<Copy>();
            std::ignore          = only;
            return gr::makeDeviceGroup(std::move(inner));
        };

        auto first  = deviceGroup();
        auto second = deviceGroup();
        expect(first.has_value() && second.has_value());
        if (!first || !second) {
            return;
        }

        gr::Graph outer;
        auto&     src = outer.emplaceBlock<Src>({{"n_samples_max", kSamples}, {"mark_tag", false}});
        auto&     snk = outer.emplaceBlock<Snk>({{"n_samples_expected", kSamples}, {"log_samples", true}});

        const std::string firstName(outer.addBlock(std::move(first->block))->uniqueName());
        const std::string secondName(outer.addBlock(std::move(second->block))->uniqueName());
        expect(outer.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(firstName), first->inputs.at(0), gr::undefined_size, 0, "src->A").has_value());
        expect(outer.emplaceEdge(std::string_view(firstName), first->outputs.at(0), std::string_view(secondName), second->inputs.at(0), gr::undefined_size, 0, "A->B").has_value());
        expect(outer.emplaceEdge(std::string_view(secondName), second->outputs.at(0), std::string_view(snk.unique_name), "in", gr::undefined_size, 0, "B->sink").has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(outer)).has_value());
        expect(sched.runAndWait().has_value()) << "two chained device groups must run to completion";
        expect(eq(snk._samples.size(), static_cast<std::size_t>(kSamples))) << "every sample must cross both groups";

        // the group-to-group edge is what used to be forced onto the host: the wrapper carried no domain outward, so
        // however the members were placed, the hop between two device groups was a host hop
        const auto edgeToB = std::ranges::find_if(sched.graph().edges(), [](const gr::Edge& edge) { return edge.name() == "A->B"; });
        expect(edgeToB != sched.graph().edges().end()) << "the group-to-group edge must still be in the graph";
        if (edgeToB != sched.graph().edges().end()) {
            expect(edgeToB->_domain.isDevice()) << std::format("the edge between two '{}' groups must be a device edge, got kind '{}' backend '{}'", *served, edgeToB->_domain.kind, edgeToB->_domain.backend);
        }
    };

    "insertion handles more than one boundary port per side"_test = [] {
        // two transfers of one type, both unnamed, would export the same port name and the group would be refused;
        // each therefore gets its own name. Structure only -- two independent members are not a runnable chain.
        gr::Graph inner;
        std::ignore = inner.emplaceBlock<Copy>();
        std::ignore = inner.emplaceBlock<Copy>();

        const auto group = gr::basic::makeBridgedDeviceGroup<float>(std::move(inner));
        expect(group.has_value()) << [&] { return group ? std::string{} : group.error().message; };
        if (!group) {
            return;
        }
        expect(eq(group->inputs.size(), 2UZ)) << "each unclaimed input must get its own transfer";
        expect(eq(group->outputs.size(), 2UZ)) << "and each unclaimed output too";
        expect(group->inputs.at(0) != group->inputs.at(1)) << "the two exported input names must differ";
    };

    "a group that gains a second device domain after construction still says so at start"_test = [] {
        // makeDeviceGroup refuses two device domains, but graph() hands out a mutable reference afterwards -- so the
        // invariant has to be a property of a running group, not of one construction path
        gr::Graph inner;
        std::ignore = inner.emplaceBlock<Copy>({{"compute_domain", std::string("gpu:sycl")}});

        auto group = gr::makeDeviceGroup(std::move(inner));
        expect(group.has_value()) << "one device domain must construct fine";
        if (!group) {
            return;
        }
        auto* wrapper = static_cast<gr::DeviceGroupWrapper*>(group->block.get());
        // bypasses the construction-time check: graph() hands out a mutable reference
        std::ignore = wrapper->blockRef().graph().emplaceBlock<Copy>({{"compute_domain", std::string("gpu:cuda")}});

        gr::log::HistoryLoggerBackend capture;
        auto* const                   previous = gr::log::setBackend(&capture);
        wrapper->start();
        wrapper->stop();
        std::ignore = gr::log::setBackend(previous);

        bool           sawRefusal = false;
        constexpr auto matcher    = [](const gr::log::LogRecord& record, void* user) noexcept {
            if (std::string_view(record.text).contains("at most one device compute_domain")) {
                *static_cast<bool*>(user) = true;
            }
        };
        std::ignore = capture.drain(matcher, &sawRefusal);
        expect(sawRefusal) << "a group that reached two device domains after construction must say so at start, not run on silently";
    };

    "an inserted group carries the same samples as the hand-wired one"_test = [] {
        gr::Graph inner;
        auto&     first  = inner.emplaceBlock<Copy>();
        auto&     second = inner.emplaceBlock<Copy>();
        expect(inner.connect(first, "out", second, "in").has_value());
        auto group = gr::basic::makeBridgedDeviceGroup<float>(std::move(inner));
        expect(group.has_value()) << [&] { return group ? std::string{} : group.error().message; };
        if (!group) {
            return;
        }
        const std::vector<float> inserted = runGroup(std::move(group.value()));
        const std::vector<float> manual   = runVertical("host", 2UZ);
        expect(eq(inserted.size(), manual.size())) << "insertion must not change how much data flows";
        expect(std::ranges::equal(inserted, manual)) << "insertion must not change the data";
    };

    return 0;
}
