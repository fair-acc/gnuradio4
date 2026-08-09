#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <print>
#include <vector>

#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

#include "device_dsp_chain.hpp"
#include <gnuradio-4.0/testing/DeviceExpectation.hpp>

#include "device_test_helpers.hpp"

/*
 * What the shared chain computes, on the host and on every device this build serves. How fast it computes it
 * is bm_DeviceDspChain's question, not this file's.
 */

using namespace gr::dsp::demo;
using gr::test::servedDomains;

int main() {
    using namespace boost::ut;

    std::ignore = gr::device::registerSyclRuntime();
    std::ignore = gr::test::requireHostSycl();

    "the chain computes the filter it is supposed to compute"_test = [] {
        const ChainRun host = runChainOn("host", 256U, 4096U);
        expect(gt(host.samples.size(), 0UZ)) << "the host arm has to produce something before it can be an oracle";

        const std::size_t transient = kTaps.size() - 1UZ; // the outputs the overlap discards, so output m is y[m + K - 1]
        bool              matches   = true;
        for (std::size_t m = 0UZ; m < host.samples.size(); ++m) {
            matches = matches && std::abs(host.samples[m] - std::abs(1.f - 0.5f * static_cast<float>(m + transient))) < 1e-3f;
        }
        expect(matches) << "a ramp through {1, -2, 0.5} is 1 - n/2 in closed form, and the magnitude stage takes it positive";
    };

    "the same source runs unchanged on every served device"_test = [] {
        const ChainRun host = runChainOn("host", 256U, 4096U);

        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            ChainRun          onDevice;
            const std::size_t refusals = gr::test::deviceRefusalsDuring([&] { onDevice = runChainOn(domain, 256U, 4096U); });

            expect(eq(refusals, 0UZ)) << std::format("'{}' must reach the kernel, or the comparison below proves nothing", domain);
            expect(eq(onDevice.samples.size(), host.samples.size())) << std::format("'{}' produced a different number of samples", domain);
            expect(std::ranges::equal(onDevice.samples, host.samples)) << std::format("'{}' must return exactly what the host returns from the same source", domain);
        }
    };

    "a three-stage chain runs its middle filter without a barrier and returns the same samples"_test = [] {
        const ChainRun host = runTwoFirChainOn("host", 256U, 4096U);

        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            gr::device::DeviceContext* const ctx = gr::device::DeviceContextRegistry::instance().tryResolve(domain);
            if (ctx == nullptr) {
                continue;
            }

            ChainRun          onDevice;
            const std::size_t refusals = gr::test::deviceRefusalsDuring([&] { onDevice = runTwoFirChainOn(domain, 256U, 4096U); });
            expect(eq(refusals, 0UZ)) << std::format("'{}' must reach the kernel, or the comparison below proves nothing", domain);
            expect(std::ranges::equal(onDevice.samples, host.samples)) << std::format("'{}' must return exactly what the host returns from the same source", domain);

            // what the third stage costs: it sits interior, device memory in and out, so its view-form body is
            // enqueued and never awaited. The only barrier it may add is its own teardown drain -- one for the run.
            const auto barriersOf = [&](auto runChain) {
                const std::uint64_t before = ctx->syncCount();
                std::ignore                = runChain();
                return ctx->syncCount() - before;
            };
            const std::uint64_t twoStage   = barriersOf([&] { return runChainOn(domain, 256U, 4096U); });
            const std::uint64_t threeStage = barriersOf([&] { return runTwoFirChainOn(domain, 256U, 4096U); });
            std::println("  '{}' device barriers: 2 device stages = {}, 3 device stages = {}", domain, twoStage, threeStage);

            // Deferral needs memory the host may NOT read, and a SYCL CPU device has none: AdaptiveCpp's host
            // backend hands back the same pointer for a device allocation as for a host one, so `isDeviceOnly` is
            // false there -- correctly, because that memory IS host memory. Ask the context rather than the domain
            // name, so the assertion holds for whatever backend serves this domain.
            gr::device::DeviceBuffer probe         = ctx->allocate(64UZ, alignof(std::max_align_t), gr::device::Residency::devicePtr);
            const bool               hasDeviceOnly = probe && ctx->isDeviceOnly(probe.devicePointer<std::byte>());
            ctx->deallocate(probe);
            if (hasDeviceOnly) {
                expect(le(threeStage, twoStage + 1UZ)) << std::format("'{}': an interior filter may cost its teardown drain and nothing else, got {} -> {}", domain, twoStage, threeStage);
            }
        }
    };

    "a filter that submits its own kernel returns the same samples"_test = [] {
        const ChainRun host = runChainOn<DirectFirSycl<float>>("host", 256U, 4096U);

        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            ChainRun          onDevice;
            const std::size_t refusals = gr::test::deviceRefusalsDuring([&] { onDevice = runChainOn<DirectFirSycl<float>>(domain, 256U, 4096U); });

            expect(eq(refusals, 0UZ)) << std::format("'{}' must reach the hatch", domain);
            expect(std::ranges::equal(onDevice.samples, host.samples)) << std::format("'{}' must return through its own kernel exactly what the host body returns", domain);
        }
    };

    "a per-frame reduction gives the same figures on every served device"_test = [] {
        static constexpr gr::Size_t kFrame    = 64U;
        static constexpr gr::Size_t kNSamples = 4096U;

        const auto runRms = [](std::string_view domain) {
            using namespace gr::testing;
            gr::Graph flow({{"auto_size_edges_to_chunks", true}});
            auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kNSamples}, {"mark_tag", false}});
            auto&     rms    = flow.emplaceBlock<FrameRms<float>>({{"gr:compute_domain", std::string(domain)}, {"input_chunk_size", kFrame}, {"output_chunk_size", gr::Size_t(1)}, {"stride", kFrame}});
            auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});
            expect(flow.connect<"out", "in">(source, rms).has_value());
            expect(flow.connect<"out", "in">(rms, sink).has_value());
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            gr::test::runAbsorbingRefusal(sched);
            std::vector<float> figures(sink._samples.size());
            for (std::size_t i = 0UZ; i < figures.size(); ++i) {
                figures[i] = sink._samples[i];
            }
            return figures;
        };

        const std::vector<float> host = runRms("host");
        expect(eq(host.size(), static_cast<std::size_t>(kNSamples / kFrame))) << "one figure per frame, and no frame left behind";

        // the source ramps, so frame f covers samples [f*N, (f+1)*N) and its RMS is known in closed form
        bool matchesClosedForm = true;
        for (std::size_t f = 0UZ; f < host.size(); ++f) {
            double sumOfSquares = 0.0;
            for (std::size_t i = 0UZ; i < kFrame; ++i) {
                const double sample = static_cast<double>(f * kFrame + i);
                sumOfSquares += sample * sample;
            }
            matchesClosedForm = matchesClosedForm && std::abs(static_cast<double>(host[f]) - std::sqrt(sumOfSquares / kFrame)) < 1e-2;
        }
        expect(matchesClosedForm) << "the host arm has to be right before it can be the oracle";

        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            const std::vector<float> onDevice = runRms(domain);
            expect(eq(onDevice.size(), host.size())) << std::format("'{}' produced a different number of figures", domain);
            expect(std::ranges::equal(onDevice, host)) << std::format("'{}' must reduce each frame to what the host reduces it to", domain);
        }
    };

    "a fixed number of channels reaches a device as one work item per sample"_test = [] {
        static constexpr std::size_t kChannels = 4UZ;
        static constexpr gr::Size_t  kNSamples = 1024U;
        const std::vector<float>     kGains    = {1.f, 2.f, 3.f, 4.f};

        const auto runChanneliser = [&](std::string_view domain) {
            using namespace gr::testing;
            gr::Graph flow({{"auto_size_edges_to_chunks", true}});
            auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kNSamples}, {"mark_tag", false}});
            auto&     dut    = flow.emplaceBlock<Channeliser<float, kChannels>>({{"gr:compute_domain", std::string(domain)}});
            dut.gains.assign(kGains.begin(), kGains.end());
            std::vector<TagSink<float, ProcessFunction::USE_PROCESS_ONE>*> sinks;
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                sinks.push_back(&flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}}));
            }
            expect(flow.connect<"out", "in">(source, dut).has_value());
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                expect(flow.connect(dut, gr::PortDefinition{std::format("out#{}", channel)}, *sinks[channel], gr::PortDefinition{"in"}).has_value());
            }
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            gr::test::runAbsorbingRefusal(sched);

            std::vector<std::vector<float>> perChannel(kChannels);
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                perChannel[channel].resize(sinks[channel]->_samples.size());
                for (std::size_t i = 0UZ; i < perChannel[channel].size(); ++i) {
                    perChannel[channel][i] = sinks[channel]->_samples[i];
                }
            }
            return perChannel;
        };

        const auto host        = runChanneliser("host");
        bool       hostIsRight = true;
        for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
            hostIsRight = hostIsRight && host[channel].size() == static_cast<std::size_t>(kNSamples);
            for (std::size_t i = 0UZ; hostIsRight && i < host[channel].size(); ++i) {
                hostIsRight = std::abs(host[channel][i] - static_cast<float>(i) * kGains[channel]) < 1e-3f;
            }
        }
        expect(hostIsRight) << "each channel carries the source scaled by its own gain";

        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            const auto onDevice = runChanneliser(domain);
            bool       matches  = true;
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                matches = matches && std::ranges::equal(onDevice[channel], host[channel]);
            }
            expect(matches) << std::format("'{}' must fill every channel with what the host fills it with", domain);
        }
    };

    "a device-resident Channeliser output elides its per-channel scratch"_test = [] {
        const auto gpuDomain = gr::test::firstServedDomain({"gpu:sycl"});
        if (!gpuDomain) {
            // a lane told to require this domain must exercise it, not report a pass having asserted nothing
            expect(!gr::testing::deviceDomainRequired("gpu:sycl")) << "GR4_REQUIRE_DEVICE names gpu:sycl, so this lane must serve it";
            return; // the elision only differs from the copy-back path once memory can genuinely be device-resident
        }
        static constexpr std::size_t kChannels = 4UZ;
        static constexpr gr::Size_t  kNSamples = 1024U;
        const std::vector<float>     kGains    = {1.f, 2.f, 3.f, 4.f};

        // identical to `runChanneliser` above except for one thing: the domain given to the four out#c -> in
        // edges. Left default, they cross to the host on a boundary edge (HostOnly access, not device-accessible).
        // Set to a shared GPU domain, ExecutionStrategy::stageOutputPort finds every channel already
        // device-accessible and returns the edges' own pointers instead of allocating a channel-major scratch block.
        const auto runChanneliser = [&](gr::EdgeParameters outputEdge) {
            using namespace gr::testing;
            gr::Graph flow({{"auto_size_edges_to_chunks", true}});
            auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kNSamples}, {"mark_tag", false}});
            auto&     dut    = flow.emplaceBlock<Channeliser<float, kChannels>>({{"gr:compute_domain", std::string(*gpuDomain)}});
            dut.gains.assign(kGains.begin(), kGains.end());
            std::vector<TagSink<float, ProcessFunction::USE_PROCESS_ONE>*> sinks;
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                sinks.push_back(&flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}}));
            }
            expect(flow.connect<"out", "in">(source, dut).has_value());
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                expect(flow.connect(dut, gr::PortDefinition{std::format("out#{}", channel)}, *sinks[channel], gr::PortDefinition{"in"}, outputEdge).has_value());
            }
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            gr::test::runAbsorbingRefusal(sched);

            std::vector<std::vector<float>> perChannel(kChannels);
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                perChannel[channel].assign(sinks[channel]->_samples.begin(), sinks[channel]->_samples.end());
            }
            return perChannel;
        };

        const auto resident = runChanneliser(gr::EdgeParameters{.domain = gr::ComputeDomain::gpu_shared()});
        const auto hostSide = runChanneliser(gr::EdgeParameters{});

        // the resident arm takes the elided branch, so a channel pointed at the wrong place shows up here as
        // wrong numbers. that the allocation is genuinely skipped is a throughput claim, and bm_DeviceDispatch
        // is where it is measured.
        expect(!resident.empty() && !resident[0].empty()) << "the resident arm must produce samples for the comparison to mean anything";
        bool matches = true;
        for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
            matches = matches && std::ranges::equal(resident[channel], hostSide[channel]);
        }
        expect(matches) << "the elided and copy-back paths must fill every channel with the same values";
    };

    "a combiner gathers its input channels the same way on every served device"_test = [] {
        static constexpr std::size_t kChannels = 4UZ;
        static constexpr gr::Size_t  kNSamples = 4096U;
        const std::vector<float>     kWeights  = {1.f, 2.f, 3.f, 4.f};

        const auto runCombiner = [&](std::string_view domain) {
            using namespace gr::testing;
            gr::Graph flow({{"auto_size_edges_to_chunks", true}});
            auto&     split = flow.emplaceBlock<Channeliser<float, kChannels>>();
            split.gains.assign(kWeights.begin(), kWeights.end());
            auto& source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kNSamples}, {"mark_tag", false}});
            auto& dut    = flow.emplaceBlock<Combiner<float, kChannels>>({{"gr:compute_domain", std::string(domain)}});
            dut.weights.assign(kWeights.begin(), kWeights.end());
            auto& sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

            expect(flow.connect<"out", "in">(source, split).has_value());
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                expect(flow.connect(split, gr::PortDefinition{std::format("out#{}", channel)}, dut, gr::PortDefinition{std::format("in#{}", channel)}).has_value());
            }
            expect(flow.connect<"out", "in">(dut, sink).has_value());

            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            gr::test::runAbsorbingRefusal(sched);
            return std::vector<float>(sink._samples.begin(), sink._samples.end());
        };

        const std::vector<float> host = runCombiner("host");
        expect(!host.empty()) << "the host arm must produce samples for the device arms to be compared against";

        for (std::string_view domain : gr::test::servedDomains()) {
            const std::vector<float> onDevice = runCombiner(domain);
            expect(eq(onDevice.size(), host.size())) << std::format("domain '{}' produced a different number of samples", domain);
            bool matches = onDevice.size() == host.size();
            for (std::size_t i = 0UZ; matches && i < host.size(); ++i) {
                matches = std::abs(onDevice[i] - host[i]) <= 1e-3f * std::max(1.f, std::abs(host[i]));
            }
            expect(matches) << std::format("domain '{}' must gather the same channels into the same sums as the host", domain);
        }
    };

    "a run-time channel count is refused by name, not silently run on the host"_test = [] {
        const auto deviceDomain = gr::test::firstServedDomain({"gpu:sycl", "host:sycl"});
        if (!deviceDomain) {
            expect(!gr::testing::deviceDomainRequired("host:sycl")) << "GR4_REQUIRE_DEVICE names host:sycl, so this lane must serve it";
            return; // nothing serves a device here, so there is no gate to exercise
        }
        using namespace gr::testing;
        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     dut  = flow.emplaceBlock<DynamicCombiner<float>>({{"gr:compute_domain", std::string(*deviceDomain) + "!"}, {"n_inputs", gr::Size_t(2)}});
        auto&     sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});
        for (std::size_t channel = 0UZ; channel < 2UZ; ++channel) {
            auto& source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(64)}, {"mark_tag", false}});
            expect(flow.connect(source, gr::PortDefinition{"out"}, dut, gr::PortDefinition{std::format("in#{}", channel)}).has_value());
        }
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        gr::test::runAbsorbingRefusal(sched);

        expect(sched.state() == gr::lifecycle::State::ERROR) << "a run-time channel count must stop the run, not be quietly demoted to the host";
        expect(sink._samples.empty()) << "a refused block must not have produced anything";
    };

    "without the marker the same block warns once and runs on the host"_test = [] {
        const auto deviceDomain = gr::test::firstServedDomain({"gpu:sycl", "host:sycl"});
        if (!deviceDomain) {
            expect(!gr::testing::deviceDomainRequired("host:sycl")) << "GR4_REQUIRE_DEVICE names host:sycl, so this lane must serve it";
            return; // nothing serves a device here, so there is no gate to exercise
        }
        using namespace gr::testing;
        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     dut  = flow.emplaceBlock<DynamicCombiner<float>>({{"gr:compute_domain", std::string(*deviceDomain)}, {"n_inputs", gr::Size_t(2)}});
        auto&     sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});
        for (std::size_t channel = 0UZ; channel < 2UZ; ++channel) {
            auto& source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(64)}, {"mark_tag", false}});
            expect(flow.connect(source, gr::PortDefinition{"out"}, dut, gr::PortDefinition{std::format("in#{}", channel)}).has_value());
        }
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        gr::test::runAbsorbingRefusal(sched);

        expect(sched.state() != gr::lifecycle::State::ERROR) << "without '!' an unreachable domain is a warning, not a stop";
        expect(!sink._samples.empty()) << "the block must have run on the host";
    };

    "a run-time channel count with a hatch is not caught by that refusal"_test = [] {
        const auto deviceDomain = gr::test::firstServedDomain({"gpu:sycl", "host:sycl"});
        if (!deviceDomain) {
            expect(!gr::testing::deviceDomainRequired("host:sycl")) << "GR4_REQUIRE_DEVICE names host:sycl, so this lane must serve it";
            return; // nothing serves a device here, so there is no gate to exercise
        }
        using namespace gr::testing;
        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     dut  = flow.emplaceBlock<DynamicCombinerWithHatch<float>>({{"gr:compute_domain", std::string(*deviceDomain) + "!"}, {"n_inputs", gr::Size_t(2)}});
        auto&     sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});
        for (std::size_t channel = 0UZ; channel < 2UZ; ++channel) {
            auto& source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(64)}, {"mark_tag", false}});
            expect(flow.connect(source, gr::PortDefinition{"out"}, dut, gr::PortDefinition{std::format("in#{}", channel)}).has_value());
        }
        expect(flow.connect<"out", "in">(dut, sink).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        gr::test::runAbsorbingRefusal(sched);

        // the refusal names the hatch as the way out, so a block that already owns one must not be caught by it
        expect(sched.state() != gr::lifecycle::State::ERROR) << "a block owning a processBulk(ctx, ...) hatch must survive the run-time-channel-count gate";
        expect(!sink._samples.empty()) << "the hatch must have run and produced samples";
    };

    "a correlator gives every lag the same value on every served device"_test = [] {
        static constexpr gr::Size_t kLength   = 32U; // reference length
        static constexpr gr::Size_t kLags     = 64U; // lags computed per window
        static constexpr gr::Size_t kNSamples = 4096U;

        const std::vector<float> reference = [] {
            std::vector<float> pattern(kLength);
            for (std::size_t k = 0UZ; k < kLength; ++k) {
                pattern[k] = std::sin(0.4f * static_cast<float>(k));
            }
            return pattern;
        }();

        const auto runCorrelator = [&](std::string_view domain) {
            using namespace gr::testing;
            gr::Graph flow({{"auto_size_edges_to_chunks", true}});
            auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kNSamples}, {"mark_tag", false}});
            auto&     dut    = flow.emplaceBlock<Correlator<float>>({{"gr:compute_domain", std::string(domain)}, //
                       {"input_chunk_size", kLags + kLength - 1U}, {"output_chunk_size", kLags}, {"stride", kLags}});
            auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});
            dut.reference.assign(reference.begin(), reference.end());
            expect(flow.connect<"out", "in">(source, dut).has_value());
            expect(flow.connect<"out", "in">(dut, sink).has_value());
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            gr::test::runAbsorbingRefusal(sched);
            std::vector<float> lags(sink._samples.size());
            for (std::size_t i = 0UZ; i < lags.size(); ++i) {
                lags[i] = sink._samples[i];
            }
            return lags;
        };

        const std::vector<float> host = runCorrelator("host");
        expect(gt(host.size(), 0UZ));

        // the source ramps, so lag l of the first window correlates the reference against samples [l, l+K)
        bool matchesDefinition = true;
        for (std::size_t lag = 0UZ; lag < std::min(host.size(), std::size_t{16}); ++lag) {
            double expected = 0.0;
            for (std::size_t k = 0UZ; k < kLength; ++k) {
                expected += static_cast<double>(reference[k]) * static_cast<double>(lag + k);
            }
            matchesDefinition = matchesDefinition && std::abs(static_cast<double>(host[lag]) - expected) < 1e-2;
        }
        expect(matchesDefinition) << "the host arm must compute the correlation before it can be the oracle";

        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            const std::vector<float> onDevice = runCorrelator(domain);
            expect(eq(onDevice.size(), host.size())) << std::format("'{}' produced a different number of lags", domain);
            // a correlation sums thirty-two products; the device may reassociate them, so the tolerance is the
            // arithmetic's, not the algorithm's -- unlike the pointwise chains above, which really are bit-equal
            // a device may contract a*b+c into one rounding, so it is not bit-equal to the host and is in fact
            // closer to the exact value; both are therefore measured against a double-precision reference
            double worst = 0.0;
            for (std::size_t lag = 0UZ; lag < std::min<std::size_t>(256UZ, host.size()); ++lag) {
                double exact = 0.0;
                for (std::size_t k = 0UZ; k < kLength; ++k) {
                    exact += static_cast<double>(reference[k]) * static_cast<double>(lag + k);
                }
                worst = std::max(worst, std::abs(static_cast<double>(onDevice[lag]) - exact) / std::max(1.0, std::abs(exact)));
            }
            expect(lt(worst, 1e-5)) << std::format("'{}' departs from the exact correlation by {} relative", domain, worst);
        }
    };
}
