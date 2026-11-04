#include <boost/ut.hpp>

#include <cstddef>
#include <memory>
#include <string>

#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

using namespace gr::testing;

int main() {
    using namespace boost::ut;

    const auto _graphSweepDomains = std::make_shared<std::size_t>(0UZ);

    "the sweep always serves the host, and nothing it was configured for is missing"_test = [] {
        const auto domains = servedDomains();
        expect(ge(domains.size(), 1UZ));
        expect(eq(domains.front(), kHostDomain));
        expect(missingRequiredDomains().empty()) << "a domain this run was configured for must be served";
    };

    // the reporting path is exercised through a DomainContext directly: a deliberate failure inside a
    // `_domain_test` would fail this test rather than being inspected
    "a failed expectation is reported with the line that raised it"_test = [] {
        gr::device::DeviceContextCpu host;
        DomainContext                context{kHostDomain, host};
        context.launch([](const DeviceTestHandle& device) { gr::testing::expect(device, false, "deliberate {} failure", 42); });

        const std::vector<std::string> failures = context.failures();
        expect(eq(failures.size(), 1UZ)) << "one record for one failed expectation";
        if (!failures.empty()) {
            expect(failures.front().contains("qa_DeviceTestHelper.cpp:")) << failures.front();
            expect(failures.front().contains("deliberate 42 failure")) << failures.front();
        }
    };

    "a satisfied expectation reports nothing"_test = [] {
        gr::device::DeviceContextCpu host;
        DomainContext                context{kHostDomain, host};
        context.launch([](const DeviceTestHandle& device) { gr::testing::expect(device, true, "never rendered"); });
        expect(context.failures().empty());
    };

    "the message-free form reports without inventing a location"_test = [] {
        gr::device::DeviceContextCpu host;
        DomainContext                context{kHostDomain, host};
        context.launch([](const DeviceTestHandle& device) { gr::testing::expect(device, false); });

        const std::vector<std::string> failures = context.failures();
        expect(eq(failures.size(), 1UZ));
        if (!failures.empty()) {
            expect(failures.front().contains("expectation failed")) << failures.front();
            expect(!failures.front().contains(".cpp:")) << "a site captured here would name the helper, not the caller";
        }
    };

    "failed() lets a kernel stop after the first complaint"_test = [] {
        gr::device::DeviceContextCpu host;
        DomainContext                context{kHostDomain, host};
        context.launch([](const DeviceTestHandle& device) {
            gr::testing::expect(device, false, "first");
            if (device.failed()) {
                return;
            }
            gr::testing::expect(device, false, "second");
        });
        expect(eq(context.failures().size(), 1UZ)) << "the second expectation must not have been reached";
    };

    "memory handed out by the context round-trips through a kernel"_domain_test = [](auto& ctx) {
        int* cell = ctx.template alloc<int>(1UZ);
        cell[0]   = 7;
        ctx.launch([cell](const DeviceTestHandle& device) {
            cell[0] *= 6;
            gr::testing::expect(device, cell[0] == 42, "the kernel computed {}", cell[0]);
        });
        boost::ut::expect(boost::ut::eq(cell[0], 42)) << "and the host sees what the kernel wrote";
    } | kAllDomains;

    "a graph sweep gives each domain its own fixture"_domain_test = [](auto& fixture, std::string_view domain) {
        auto& source = fixture.template emplace<gr::testing::ConstantSource<float>>({{"n_samples_max", gr::Size_t(16)}});
        auto& sink   = fixture.template emplace<gr::testing::CountingSink<float>>();
        expect(fixture.template connect<"out", "in">(source, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink.count, 16UZ)) << "the fixture keeps the blocks alive per domain";
    } | gr::testing::overGraphs(kAllDomains);

    "launchRange visits every index exactly once"_domain_test = [](auto& ctx) {
        constexpr std::size_t kCount = 8UZ;
        int*                  hits   = ctx.template alloc<int>(kCount);
        for (std::size_t i = 0UZ; i < kCount; ++i) {
            hits[i] = 0;
        }
        ctx.launchRange(kCount, [hits](const DeviceTestHandle&, std::size_t i) { hits[i] += 1; });

        int total = 0;
        for (std::size_t i = 0UZ; i < kCount; ++i) {
            total += hits[i];
        }
        boost::ut::expect(boost::ut::eq(total, static_cast<int>(kCount)));
    } | kAllDomains;

    return 0;
}
