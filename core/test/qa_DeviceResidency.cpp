#include <boost/ut.hpp>

#include <tuple>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <algorithm>
#include <ranges>
#include <string>
#include <vector>

#include "device_test_helpers.hpp"

namespace gr::residency {

/// an ordinary device-eligible block: `const noexcept processOne`, nothing device-specific about it
struct Gain : gr::Block<Gain> {
    using Description = Doc<"Scales each sample; used to chain two device blocks back to back.">;

    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    float gain = 2.f;

    GR_MAKE_REFLECTABLE(Gain, in, out, gain);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x * gain; }
};

} // namespace gr::residency

namespace {

struct EdgeResidency {
    std::string name;
    bool        domainInterior = false; // both endpoints on one device domain: the edge never crosses to the host
    bool        deviceOnly     = false;
    bool        usesMMAP       = false;
    bool        mirrorsItself  = false; // device-only memory that mirrors its own wrap instead of double-mapping
    bool        connected  = false; // an edge that failed to connect leaves its ports on their default buffers,
                                    // so the graph still runs -- without this the test would pass on intent alone
    const void*                sourcePort = nullptr; // two edges sharing one identify a fan-out
    std::pmr::memory_resource* resource   = nullptr;
};

/// host source -> gpu -> gpu -> host sink: the middle edge is interior to one device, the outer two cross the boundary
[[nodiscard]] std::vector<EdgeResidency> runTwoDeviceBlockChain(std::string_view domain, gr::Size_t nSamples, std::vector<float>& sinkSamples) {
    using namespace gr::testing;

    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     first  = flow.emplaceBlock<gr::residency::Gain>({{"gr:compute_domain", std::string(domain)}, {"gain", 2.f}});
    auto&     second = flow.emplaceBlock<gr::residency::Gain>({{"gr:compute_domain", std::string(domain)}, {"gain", 3.f}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", nSamples}, {"log_samples", true}});

    std::ignore = flow.connect<"out", "in">(source, first);
    std::ignore = flow.connect<"out", "in">(first, second); // the interior edge
    std::ignore = flow.connect<"out", "in">(second, sink);

    gr::scheduler::Simple<> scheduler;
    std::ignore = scheduler.exchange(std::move(flow));
    std::ignore = scheduler.runAndWait();

    std::vector<EdgeResidency> residency;
    for (const gr::Edge& edge : scheduler.graph().edges()) {
        residency.push_back({.name = edge._name, .domainInterior = edge._domain.access == gr::Access::DeviceOnly, .deviceOnly = gr::isDeviceOnly(edge._dataResource), //
            .usesMMAP = gr::usesMMAP(edge._dataResource), .mirrorsItself = gr::memoryResourceCapabilities(edge._dataResource).copyWithin != nullptr, //
            .connected = edge._state == gr::Edge::EdgeState::Connected, .sourcePort = edge._sourcePort, .resource = edge._dataResource});
    }

    sinkSamples.assign(sink._samples.begin(), sink._samples.end());
    return residency;
}

/// host source -> head -> {armA, armB} -> host sinks, the three Gains on one domain: one port, two consumers
[[nodiscard]] std::vector<EdgeResidency> runDeviceFanOut(std::string_view domain, gr::Size_t nSamples, std::vector<float>& armASamples, std::vector<float>& armBSamples) {
    using namespace gr::testing;

    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     head   = flow.emplaceBlock<gr::residency::Gain>({{"gr:compute_domain", std::string(domain)}, {"gain", 2.f}});
    auto&     armA   = flow.emplaceBlock<gr::residency::Gain>({{"gr:compute_domain", std::string(domain)}, {"gain", 3.f}});
    auto&     armB   = flow.emplaceBlock<gr::residency::Gain>({{"gr:compute_domain", std::string(domain)}, {"gain", 5.f}}); // differs from armA so a crossed wiring shows up in the samples
    auto&     sinkA  = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", nSamples}, {"log_samples", true}});
    auto&     sinkB  = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", nSamples}, {"log_samples", true}});

    std::ignore = flow.connect<"out", "in">(source, head);
    std::ignore = flow.connect<"out", "in">(head, armA); // both fan-out edges leave the one device port
    std::ignore = flow.connect<"out", "in">(head, armB);
    std::ignore = flow.connect<"out", "in">(armA, sinkA);
    std::ignore = flow.connect<"out", "in">(armB, sinkB);

    gr::scheduler::Simple<> scheduler;
    std::ignore = scheduler.exchange(std::move(flow));
    std::ignore = scheduler.runAndWait();

    std::vector<EdgeResidency> residency;
    for (const gr::Edge& edge : scheduler.graph().edges()) {
        residency.push_back({.name = edge._name, .domainInterior = edge._domain.access == gr::Access::DeviceOnly, .deviceOnly = gr::isDeviceOnly(edge._dataResource), //
            .usesMMAP = gr::usesMMAP(edge._dataResource), .mirrorsItself = gr::memoryResourceCapabilities(edge._dataResource).copyWithin != nullptr,                    //
            .connected = edge._state == gr::Edge::EdgeState::Connected, .sourcePort = edge._sourcePort, .resource = edge._dataResource});
    }

    armASamples.assign(sinkA._samples.begin(), sinkA._samples.end());
    armBSamples.assign(sinkB._samples.begin(), sinkB._samples.end());
    return residency;
}

[[nodiscard]] std::vector<EdgeResidency> edgesLeavingTheFannedOutPort(const std::vector<EdgeResidency>& residency) {
    const auto sharesItsSourceWithAnother = [&residency](const EdgeResidency& candidate) {
        return std::ranges::count_if(residency, [&candidate](const EdgeResidency& other) { return other.sourcePort == candidate.sourcePort; }) > 1L;
    };
    return residency | std::views::filter(sharesItsSourceWithAnother) | std::ranges::to<std::vector>();
}

} // namespace

// AdaptiveCpp aborts if a kernel is launched while Boost.UT runs suites from ~runner, so tests run from main (gotcha G10)
int main() {
    using namespace boost::ut;

    std::ignore = gr::device::registerSyclRuntime(); // the domains below are only served once the runtime registers

    "a host-only graph puts nothing in device-only memory"_test = [] {
        constexpr gr::Size_t kN = 64U;
        std::vector<float>   samples;
        const auto           residency = runTwoDeviceBlockChain("host", kN, samples);

        expect(!residency.empty());
        expect(std::ranges::all_of(residency, [](const EdgeResidency& e) { return e.connected; })) << "every edge must actually connect";
        expect(std::ranges::none_of(residency, [](const EdgeResidency& e) { return e.domainInterior || e.deviceOnly; })) << "a host graph must never mark an edge interior to a device, nor allocate device-only memory";
        expect(eq(samples.size(), std::size_t{kN}));
        expect(std::ranges::all_of(std::views::iota(0UZ, samples.size()), [&samples](std::size_t i) { return samples[i] == static_cast<float>(i) * 6.f; })) << "gain 2 then gain 3";
    };

    "an edge between two blocks on the same device is interior, the boundary edges are not"_test = [] {
        const auto gpuDomain = gr::test::firstServedDomain({"gpu:sycl"});
        if (!gpuDomain) {
            return;
        }

        constexpr gr::Size_t kN = 64U;
        std::vector<float>   samples;
        const auto           residency = runTwoDeviceBlockChain(*gpuDomain, kN, samples);

        expect(eq(residency.size(), 3UZ)) << "source->first, first->second, second->sink";
        expect(std::ranges::all_of(residency, [](const EdgeResidency& e) { return e.connected; })) << "every edge must actually connect -- a device-only buffer the ports could not take would silently fall back to host memory";
        const auto interior = std::ranges::count_if(residency, [](const EdgeResidency& e) { return e.domainInterior; });
        expect(eq(interior, 1L)) << "exactly the middle edge is interior to the device";
        expect(std::ranges::none_of(residency, [](const EdgeResidency& e) { return e.deviceOnly && !e.usesMMAP && !e.mirrorsItself; })) << "device-only memory must either double-map or mirror its own wrap, else the mirror would fault on the host";
        expect(eq(std::ranges::count_if(residency, [](const EdgeResidency& e) { return e.deviceOnly; }), 1L)) << "the interior edge holds memory the host cannot address";

        // the point of the exercise: data still arrives, having stayed on the device across the interior edge
        expect(eq(samples.size(), std::size_t{kN}));
        expect(std::ranges::all_of(std::views::iota(0UZ, samples.size()), [&samples](std::size_t i) { return samples[i] == static_cast<float>(i) * 6.f; })) << "gain 2 then gain 3, computed on the device";
    };

    "a fan-out feeds both consumers from the one buffer its source port owns"_test = [] {
        constexpr gr::Size_t kN = 64U;
        std::vector<float>   armA;
        std::vector<float>   armB;
        const auto           residency = runDeviceFanOut("host", kN, armA, armB);

        expect(eq(residency.size(), 5UZ)) << "source->head, head->armA, head->armB, armA->sinkA, armB->sinkB";
        expect(std::ranges::all_of(residency, [](const EdgeResidency& e) { return e.connected; })) << "every edge must actually connect";
        expect(eq(edgesLeavingTheFannedOutPort(residency).size(), 2UZ)) << "exactly the two arms leave one port";
        expect(std::ranges::none_of(residency, [](const EdgeResidency& e) { return e.deviceOnly; })) << "a host graph must never allocate device-only memory";

        expect(eq(armA.size(), std::size_t{kN}));
        expect(eq(armB.size(), std::size_t{kN}));
        expect(std::ranges::all_of(std::views::iota(0UZ, armA.size()), [&armA](std::size_t i) { return armA[i] == static_cast<float>(i) * 6.f; })) << "gain 2 then 3";
        expect(std::ranges::all_of(std::views::iota(0UZ, armB.size()), [&armB](std::size_t i) { return armB[i] == static_cast<float>(i) * 10.f; })) << "gain 2 then 5 -- a different factor, so crossed arms cannot pass";
    };

    "two device consumers of one device port share a single device-only buffer"_test = [] {
        const auto gpuDomain = gr::test::firstServedDomain({"gpu:sycl"});
        if (!gpuDomain) {
            return;
        }

        constexpr gr::Size_t kN = 64U;
        std::vector<float>   armA;
        std::vector<float>   armB;
        const auto           residency = runDeviceFanOut(*gpuDomain, kN, armA, armB);
        const auto           fannedOut = edgesLeavingTheFannedOutPort(residency);

        expect(eq(residency.size(), 5UZ)) << "source->head, head->armA, head->armB, armA->sinkA, armB->sinkB";
        expect(std::ranges::all_of(residency, [](const EdgeResidency& e) { return e.connected; })) << "every edge must actually connect -- a device-only buffer the ports could not take would silently fall back to host memory";
        expect(eq(fannedOut.size(), 2UZ)) << "exactly the two arms leave one port";
        expect(std::ranges::all_of(fannedOut, [](const EdgeResidency& e) { return e.deviceOnly; })) << "both arms of the fan-out stay in memory the host cannot address";
        expect(fannedOut.front().resource == fannedOut.back().resource) << "one buffer serves both consumers -- two resources would mean the data was copied";
        expect(eq(std::ranges::count_if(residency, [](const EdgeResidency& e) { return e.deviceOnly; }), 2L)) << "only the fan-out is device-only; the three boundary edges cross to the host";

        // the point of the exercise: both arms receive their own result without the stream returning to the host between blocks
        expect(eq(armA.size(), std::size_t{kN}));
        expect(eq(armB.size(), std::size_t{kN}));
        expect(std::ranges::all_of(std::views::iota(0UZ, armA.size()), [&armA](std::size_t i) { return armA[i] == static_cast<float>(i) * 6.f; })) << "gain 2 then 3, computed on the device";
        expect(std::ranges::all_of(std::views::iota(0UZ, armB.size()), [&armB](std::size_t i) { return armB[i] == static_cast<float>(i) * 10.f; })) << "gain 2 then 5, computed on the device";
    };

    "an interior edge keeps its tag buffer in host-accessible memory"_test = [] {
        const auto gpuDomain = gr::test::firstServedDomain({"gpu:sycl"});
        if (!gpuDomain) {
            return;
        }
        gr::ComputeDomain tagAxis = gr::ComputeDomain::parse(*gpuDomain);
        tagAxis.access            = gr::Access::Shared;
        // a kernel may not publish tags, so tag storage must remain host-addressable
        expect(!gr::isDeviceOnly(gr::ComputeRegistry::instance().tryResolve(tagAxis, tagAxis.user))) << "the tag axis must never resolve to device-only memory";
    };

    return 0;
}
