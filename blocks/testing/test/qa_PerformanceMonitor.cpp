#include <boost/ut.hpp>

#include <memory>
#include <string>
#include <vector>

#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/PerformanceMonitor.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace boost::ut;
using gr::testing::PerformanceMonitor;
using Fixture = gr::testing::GraphFixture<gr::scheduler::Simple<>>;
using Rates   = gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_BULK>;

struct PassOnDevice : gr::Block<PassOnDevice> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;
    GR_MAKE_REFLECTABLE(PassOnDevice, in, out);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x; }
};

const auto tests = [] {
    "a rate is the count over the interval in seconds, and 0 over an empty interval"_test = [] {
        expect(eq(gr::testing::details::perSecond(10U, 2'000'000), 5.)) << "10 items in 2 s";
        expect(eq(gr::testing::details::perSecond(3U, 500), 6000.)) << "3 items in 0.5 ms";
        expect(eq(gr::testing::details::perSecond(7U, 0), 0.));
        expect(eq(gr::testing::details::perSecond(7U, -1), 0.)) << "a clock that went back";
    };

    "a DataSet stream is counted in data sets, one rate per evaluate_perf_rate of them"_test = [] {
        Fixture fixture;
        auto&   source  = fixture.emplace<gr::testing::ConstantSource<gr::DataSet<float>>>({{"n_samples_max", gr::Size_t{50U}}});
        auto&   monitor = fixture.emplace<PerformanceMonitor<gr::DataSet<float>>>({{"evaluate_perf_rate", gr::Size_t{10U}}, {"publish_rate", 1000.f}});
        auto&   rates   = fixture.emplace<Rates>({{"log_samples", true}});
        expect(fixture.connect<"out", "in">(source, monitor).has_value());
        expect(fixture.connect<"outRate", "in">(monitor, rates).has_value());
        expect(fixture.run().has_value());

        expect(eq(rates._samples.size(), 5UZ)) << "50 data sets, one evaluation per 10";
        expect(std::ranges::all_of(rates._samples, [](double rate) { return rate > 0.; })) << "data sets per second";
        expect(eq(monitor.n_updates_rate, gr::Size_t{5U}));
    };

    "a stream slower than evaluate_perf_rate still reports, every publish_rate seconds"_test = [] {
        Fixture fixture;
        auto&   source  = fixture.emplace<gr::testing::ConstantSource<float>>({{"n_samples_max", gr::Size_t{1000U}}});
        auto&   monitor = fixture.emplace<PerformanceMonitor<float>>({{"publish_rate", 0.f}}); // the default evaluate_perf_rate, 1'000'000, is never reached
        auto&   rates   = fixture.emplace<Rates>({{"log_samples", true}});
        expect(fixture.connect<"out", "in">(source, monitor).has_value());
        expect(fixture.connect<"outRate", "in">(monitor, rates).has_value());
        expect(fixture.run().has_value());

        expect(ge(rates._samples.size(), 1UZ)) << "an interval of 0 s has elapsed at every call";
    };

    "the memory figures leave on their own ports at each evaluation"_test = [] {
        Fixture fixture;
        auto&   source   = fixture.emplace<gr::testing::ConstantSource<float>>({{"n_samples_max", gr::Size_t{3000U}}});
        auto&   monitor  = fixture.emplace<PerformanceMonitor<float>>({{"evaluate_perf_rate", gr::Size_t{1000U}}, {"publish_rate", 1000.f}});
        auto&   resident = fixture.emplace<Rates>({{"log_samples", true}});
        auto&   address  = fixture.emplace<Rates>({{"log_samples", true}});
        expect(fixture.connect<"out", "in">(source, monitor).has_value());
        expect(fixture.connect<"outRes", "in">(monitor, resident).has_value());
        expect(fixture.connect<"outVirt", "in">(monitor, address).has_value());
        expect(fixture.run().has_value());

        expect(resident._samples.size() == 3UZ && address._samples.size() == 3UZ) << "3000 samples, one evaluation per 1000";
        for (std::size_t i = 0UZ; i < std::min(resident._samples.size(), address._samples.size()); ++i) {
            expect(address._samples[i] > 0.) << "the virtual size, or the wasm memory under Emscripten";
            expect(address._samples[i] >= resident._samples[i]) << "the resident pages are part of it";
        }
    };

    "an unknown device_domain reports a zero sync rate and the item rate as before"_test = [] {
        Fixture fixture;
        auto&   source  = fixture.emplace<gr::testing::ConstantSource<float>>({{"n_samples_max", gr::Size_t{2000U}}});
        auto&   monitor = fixture.emplace<PerformanceMonitor<float>>({{"evaluate_perf_rate", gr::Size_t{1000U}}, {"publish_rate", 1000.f}, {"device_domain", std::string("gpu:absent")}});
        auto&   rates   = fixture.emplace<Rates>({{"log_samples", true}});
        auto&   syncs   = fixture.emplace<Rates>({{"log_samples", true}});
        expect(fixture.connect<"out", "in">(source, monitor).has_value());
        expect(fixture.connect<"outRate", "in">(monitor, rates).has_value());
        expect(fixture.connect<"outDeviceSyncRate", "in">(monitor, syncs).has_value());
        expect(fixture.run().has_value());

        expect(eq(rates._samples.size(), 2UZ));
        expect(eq(syncs._samples.size(), 2UZ)) << "the sync rate is reported, as 0";
        expect(std::ranges::all_of(syncs._samples, [](double rate) { return rate == 0.; }));
    };
};
const auto deviceTests = [] {
    using gr::testing::operator""_domain_test;

    "the device sync rate is the device_domain context's synchronisations per second"_domain_test = [](auto& fixture, std::string_view domain) {
        gr::device::DeviceContext* const context     = gr::device::DeviceContextRegistry::instance().tryResolve(domain);
        const std::uint64_t              syncsBefore = context->syncCount();
        auto&                            source      = fixture.template emplace<gr::testing::ConstantSource<float>>({{"n_samples_max", gr::Size_t{4000U}}});
        auto&                            pass        = fixture.template emplace<PassOnDevice>({{"gr:compute_domain", std::string(domain)}});
        auto&                            monitor     = fixture.template emplace<PerformanceMonitor<float>>({{"evaluate_perf_rate", gr::Size_t{1000U}}, {"publish_rate", 1000.f}, {"device_domain", std::string(domain)}});
        auto&                            syncs       = fixture.template emplace<Rates>({{"log_samples", true}});
        expect(fixture.template connect<"out", "in">(source, pass).has_value());
        expect(fixture.template connect<"out", "in">(pass, monitor).has_value());
        expect(fixture.template connect<"outDeviceSyncRate", "in">(monitor, syncs).has_value());
        expect(fixture.run().has_value()) << domain;

        expect(gt(context->syncCount(), syncsBefore)) << domain << ": the dispatches synchronised";
        expect(eq(syncs._samples.size(), 4UZ)) << domain;
        expect(std::ranges::all_of(syncs._samples, [](double rate) { return rate >= 0.; }) && std::ranges::any_of(syncs._samples, [](double rate) { return rate > 0.; })) << domain;
    } | gr::testing::overGraphs(gr::testing::kOffloadDomains);
};
} // namespace

int main() {
    "PerformanceMonitor"_test                               = tests;
    "PerformanceMonitor on every served device domain"_test = deviceTests;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
