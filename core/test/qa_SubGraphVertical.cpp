#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/SubGraph.hpp>
#include <gnuradio-4.0/basic/DeviceSubGraph.hpp>
#include <gnuradio-4.0/basic/TransferBlocks.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>

#include "device_test_helpers.hpp"
#include <gnuradio-4.0/testing/DeviceExpectation.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

/*
 * The vertical stack a device SubGraph is meant to carry:
 *
 *   source ─▶ SubGraph[ HostToDevice ─▶ member ─▶ member ─▶ DeviceToHost ] ─▶ sink
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

std::vector<float> runGroup(gr::SubGraphHandle group);

/// builds the group above and runs it to completion, returning what reached the sink
std::vector<float> runVertical(std::string_view memberDomain, std::size_t nMembers) {
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

    auto group = gr::makeSubGraph(std::move(inner));
    expect(group.has_value()) << [&] { return group ? std::string{} : group.error().message; };
    if (!group) {
        return {};
    }
    return runGroup(std::move(group.value()));
}

/// drives an already-built group from a source to a sink and returns what arrived
std::vector<float> runGroup(gr::SubGraphHandle group) {
    gr::Graph outer;
    auto&     src = outer.emplaceBlock<Src>({{"n_samples_max", kSamples}, {"mark_tag", false}});
    auto&     snk = outer.emplaceBlock<Snk>({{"n_samples_expected", kSamples}, {"log_samples", true}});

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

        const auto group = gr::makeSubGraph(std::move(inner));
        expect(!group.has_value()) << "two device domains in one group must be an error at construction, not a silent fallback at run time";
    };

    "the transfer blocks are ordinary members: the group exports their outer ports, not the members'"_test = [] {
        gr::Graph inner;
        auto&     h2d    = inner.emplaceBlock<gr::basic::HostToDevice<float>>();
        auto&     member = inner.emplaceBlock<Copy>();
        auto&     d2h    = inner.emplaceBlock<gr::basic::DeviceToHost<float>>();
        expect(inner.connect(h2d, "out", member, "in").has_value());
        expect(inner.connect(member, "out", d2h, "in").has_value());

        const auto group = gr::makeSubGraph(std::move(inner));
        expect(group.has_value()) << [&] { return group ? std::string{} : group.error().message; };
        if (!group) {
            return;
        }
        expect(eq(group->inputs.size(), 1UZ)) << "only HostToDevice::in is unclaimed on the input side";
        expect(eq(group->outputs.size(), 1UZ)) << "only DeviceToHost::out is unclaimed on the output side";
        expect(group->inputs.at(0).contains("HostToDevice") && group->inputs.at(0).ends_with(":in")) << "the exported name must identify the member and the port, got: " << group->inputs.at(0);
        expect(group->outputs.at(0).contains("DeviceToHost") && group->outputs.at(0).ends_with(":out")) << "the exported name must identify the member and the port, got: " << group->outputs.at(0);
    };

    "makeDeviceSubGraph builds by itself what the manual wiring above builds by hand"_test = [] {
        gr::Graph inner;
        auto&     first  = inner.emplaceBlock<Copy>();
        auto&     second = inner.emplaceBlock<Copy>();
        expect(inner.connect(first, "out", second, "in").has_value());

        const auto group = gr::basic::makeDeviceSubGraph<float>(std::move(inner), "host");
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

    "insertion handles more than one boundary port per side"_test = [] {
        // two transfers of one type, both unnamed, would export the same port name and the group would be refused;
        // each therefore gets its own name. Structure only -- two independent members are not a runnable chain.
        gr::Graph inner;
        std::ignore = inner.emplaceBlock<Copy>();
        std::ignore = inner.emplaceBlock<Copy>();

        const auto group = gr::basic::makeDeviceSubGraph<float>(std::move(inner), "host");
        expect(group.has_value()) << [&] { return group ? std::string{} : group.error().message; };
        if (!group) {
            return;
        }
        expect(eq(group->inputs.size(), 2UZ)) << "each unclaimed input must get its own transfer";
        expect(eq(group->outputs.size(), 2UZ)) << "and each unclaimed output too";
        expect(group->inputs.at(0) != group->inputs.at(1)) << "the two exported input names must differ";
    };

    "a group that gains a second device domain after construction still says so at start"_test = [] {
        // makeSubGraph refuses two device domains, but graph() hands out a mutable reference afterwards -- so the
        // invariant has to be a property of a running group, not of one construction path
        gr::Graph inner;
        std::ignore = inner.emplaceBlock<Copy>({{"compute_domain", std::string("gpu:sycl")}});

        auto group = gr::makeSubGraph(std::move(inner));
        expect(group.has_value()) << "one device domain must construct fine";
        if (!group) {
            return;
        }
        auto* wrapper = static_cast<gr::SubGraphWrapper*>(group->block.get());
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
        auto group = gr::basic::makeDeviceSubGraph<float>(std::move(inner), "host");
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
