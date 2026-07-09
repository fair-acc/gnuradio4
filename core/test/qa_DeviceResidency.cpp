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
    bool        connected  = false; // an edge that failed to connect leaves its ports on their default buffers,
                                    // so the graph still runs -- without this the test would pass on intent alone
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
            .usesMMAP = gr::usesMMAP(edge._dataResource), .connected = edge._state == gr::Edge::EdgeState::Connected});
    }

    sinkSamples.assign(sink._samples.begin(), sink._samples.end());
    return residency;
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
        expect(std::ranges::none_of(residency, [](const EdgeResidency& e) { return e.deviceOnly && !e.usesMMAP; })) << "device-only memory must be double-mapped, else the wrap mirror would fault";

        // the point of the exercise: data still arrives, having stayed on the device across the interior edge
        expect(eq(samples.size(), std::size_t{kN}));
        expect(std::ranges::all_of(std::views::iota(0UZ, samples.size()), [&samples](std::size_t i) { return samples[i] == static_cast<float>(i) * 6.f; })) << "gain 2 then gain 3, computed on the device";
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
