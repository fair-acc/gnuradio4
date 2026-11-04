#include <boost/ut.hpp>

#include <complex>
#include <format>
#include <string>
#include <string_view>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/filter/PolyphaseChannelizer.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

// the bank's numerics are proven against the algorithm in algorithm/test/qa_PolyphaseChannelizerAlgo.cpp.
// What is block-specific, and tested here, is the port collection following `n_channels`, the rate contract
// the framework drives the block by, and that a tone reaches the right channel PORT through a real graph.

int main() {
    using namespace boost::ut;

    // the device cases below must run inside main(): a namespace-scope `boost::ut::suite` executes from the
    // runner's destructor, after main() has returned and after ComputeRegistry's function-local static has
    // been destroyed -- its `_providers` map is then walked after free, which faults resolving a backend
    std::ignore = gr::device::registerSyclRuntime();
    expect(!gr::device::registerSyclRuntime() || gr::device::hostSyclIsServed()) //
        << "a build with a SYCL backend must serve 'host:sycl'; without it every device case below skips and asserts nothing";

    using namespace boost::ut;
    using gr::test::eq;
    using C = std::complex<float>;

    "the port collection follows n_channels"_test = [] {
        gr::filter::PolyphaseChannelizer<C> block({{"n_channels", gr::Size_t(8)}, {"n_taps", gr::Size_t(64)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();

        expect(eq(block.out.size(), 8UZ)) << "one channel port per channel";
        expect(eq(block.phases.size() % 8UZ, 0UZ)) << "the bank is phase-major over n_channels";
        expect(gt(block.phaseLength(), 1UZ)) << "a single tap per arm would leave a bare commutator, not a filterbank";
    };

    "the rate contract is one output set per n_channels inputs"_test = [] {
        gr::filter::PolyphaseChannelizer<C> block({{"n_channels", gr::Size_t(4)}, {"n_taps", gr::Size_t(32)}, {"outputs_per_frame", gr::Size_t(16)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();

        expect(eq(static_cast<std::size_t>(block.output_chunk_size), 16UZ)) << "each channel port gets one sample per set";
        expect(eq(static_cast<std::size_t>(block.stride), 64UZ)) << "the window advances one full set of inputs per output set";
        expect(gt(static_cast<std::size_t>(block.input_chunk_size), 64UZ)) << "the window must also carry the history the arms reach over";
    };

    "a short prototype is raised to two taps per arm"_test = [] {
        gr::filter::PolyphaseChannelizer<C> block({{"n_channels", gr::Size_t(16)}, {"n_taps", gr::Size_t(4)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();

        expect(ge(block.phaseLength(), 2UZ)) << std::format("n_taps 4 over 16 channels left {} taps per arm", block.phaseLength());
    };
    auto toneReachesItsPort = [](std::string_view domain) {
        using namespace std::string_literals;
        using namespace gr::testing;
        constexpr gr::Size_t kChannels = 4U;

        gr::Graph flow({{"auto_size_edges_to_chunks", true}});

        // one cycle of e^(j2*pi*n/4) is exactly {1, i, -1, -i}, so a cycling source IS a tone at channel 1
        std::vector<C> tone{C{1.f, 0.f}, C{0.f, 1.f}, C{-1.f, 0.f}, C{0.f, -1.f}};
        auto&          source = flow.emplaceBlock<TagSource<C>>({{"n_samples_max", gr::Size_t(8192)}, {"values", tone}, {"mark_tag", false}});
        auto&          dut    = flow.emplaceBlock<gr::filter::PolyphaseChannelizer<C>>({{"n_channels", kChannels}, {"n_taps", gr::Size_t(32)}, {"outputs_per_frame", gr::Size_t(32)}, {"gr:compute_domain", std::string(domain)}});

        expect(flow.connect(source, "out"s, dut, "in"s).has_value());

        std::vector<TagSink<C, ProcessFunction::USE_PROCESS_ONE>*> sinks;
        for (gr::Size_t k = 0U; k < kChannels; ++k) {
            sinks.push_back(std::addressof(flow.emplaceBlock<TagSink<C, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}})));
            expect(flow.connect(dut, "out#"s + std::to_string(k), *sinks[k], "in"s).has_value());
        }

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());

        std::vector<float> energy(kChannels, 0.f);
        for (gr::Size_t k = 0U; k < kChannels; ++k) {
            const auto& samples = sinks[k]->_samples;
            expect(gt(samples.size(), 0UZ)) << std::format("channel {} produced nothing", k);
            for (std::size_t i = samples.size() / 2UZ; i < samples.size(); ++i) { // skip the filter's settling
                energy[k] += std::abs(samples[i]);
            }
            energy[k] /= static_cast<float>(std::max(std::size_t{1}, samples.size() - samples.size() / 2UZ));
        }

        expect(gt(energy[1], 0.5f)) << std::format("the wanted channel averages only {:.4f}", energy[1]);
        for (gr::Size_t k = 0U; k < kChannels; ++k) {
            if (k != 1U) {
                expect(lt(energy[k], 0.2f * energy[1])) << std::format("[{}] channel {} averages {:.4f} against the wanted {:.4f}", domain, k, energy[k], energy[1]);
            }
        }
    };

    "a tone reaches its own channel port on the host"_test = [&] { toneReachesItsPort("host"); };

    // where a SYCL backend exists this goes through `processBulk(ctx, ...)` -- the only device route a dynamic
    // port collection has. Without one the domain falls back to the host and the case still has to hold.
    "a tone reaches its own channel port on a device"_test = [&] { toneReachesItsPort("host:sycl"); };
    return 0;
}
