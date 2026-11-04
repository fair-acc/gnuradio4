#include <boost/ut.hpp>

#include <algorithm>
#include <complex>
#include <format>
#include <functional>
#include <numbers>
#include <print>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/filter/HierarchicalPolyphaseChannelizer.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using C = std::complex<float>;

/// mean magnitude per output port, past the filters' settling
[[nodiscard]] std::vector<float> runTone(const std::vector<C>& pattern, gr::Size_t stage1, gr::Size_t stage2, std::string_view domain = "host") {
    using namespace std::string_literals;
    using namespace gr::testing;

    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source = flow.emplaceBlock<TagSource<C>>({{"n_samples_max", gr::Size_t(32768)}, {"values", pattern}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<gr::filter::HierarchicalPolyphaseChannelizer<C>>({{"stage1_channels", stage1}, {"stage2_channels", stage2}, {"n_taps", gr::Size_t(32)}, {"outputs_per_frame", gr::Size_t(16)}, {"gr:compute_domain", std::string(domain)}});

    boost::ut::expect(flow.connect(source, "out"s, dut, "in"s).has_value());

    std::vector<TagSink<C, ProcessFunction::USE_PROCESS_ONE>*> sinks;
    for (gr::Size_t k = 0U; k < stage1 * stage2; ++k) {
        sinks.push_back(std::addressof(flow.emplaceBlock<TagSink<C, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}})));
        boost::ut::expect(flow.connect(dut, "out#"s + std::to_string(k), *sinks[k], "in"s).has_value());
    }

    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());

    std::vector<float> energy(stage1 * stage2, 0.f);
    for (std::size_t k = 0UZ; k < energy.size(); ++k) {
        const auto&       samples = sinks[k]->_samples;
        const std::size_t from    = samples.size() / 2UZ;
        for (std::size_t i = from; i < samples.size(); ++i) {
            energy[k] += std::abs(samples[i]);
        }
        energy[k] /= static_cast<float>(std::max(std::size_t{1}, samples.size() - from));
    }
    return energy;
}
} // namespace

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

    "the port count is the product of the stages"_test = [] {
        gr::filter::HierarchicalPolyphaseChannelizer<C> block({{"stage1_channels", gr::Size_t(4)}, {"stage2_channels", gr::Size_t(8)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();

        expect(eq(block.out.size(), 32UZ)) << "4 x 8 channels";
        expect(gt(block.phaseLength1(), 1UZ)) << "stage 1 needs more than a bare commutator";
        expect(gt(block.phaseLength2(), 1UZ)) << "stage 2 needs more than a bare commutator";
    };

    "each stage designs its own prototype"_test = [] {
        gr::filter::HierarchicalPolyphaseChannelizer<C> block({{"stage1_channels", gr::Size_t(2)}, {"stage2_channels", gr::Size_t(8)}, {"n_taps", gr::Size_t(32)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();

        // a wider stage needs at least two taps per arm, so its bank is the larger of the two
        expect(gt(block.phases_stage2.size(), 0UZ));
        expect(gt(block.phases_stage1.size(), 0UZ));
        expect(neq(block.phaseLength1(), block.phaseLength2())) << "different channel counts must give different arm lengths";
    };

    "a tone inside one channel concentrates in one port"_test = [] {
        // fs/16 sits well inside stage-1 channel 0 AND inside its stage-2 channel 0, so it should not straddle
        // a boundary. fs/4 deliberately is NOT used here -- see the next case.
        std::vector<C> tone(16UZ);
        for (std::size_t n = 0UZ; n < tone.size(); ++n) {
            const float angle = 2.f * std::numbers::pi_v<float> * static_cast<float>(n) / 16.f;
            tone[n]           = C{std::cos(angle), std::sin(angle)};
        }
        const auto energy = runTone(tone, 2U, 2U);

        std::size_t best  = 0UZ;
        float       bestE = -1.f;
        float       total = 0.f;
        for (std::size_t k = 0UZ; k < energy.size(); ++k) {
            total += energy[k];
            if (energy[k] > bestE) {
                bestE = energy[k];
                best  = k;
            }
        }
        expect(gt(bestE, 0.f)) << "no port carried anything";
        expect(gt(bestE, 0.5f * total)) << std::format("the strongest port (#{}) holds {:.4f} of {:.4f}", best, bestE, total);

        std::string spread;
        for (std::size_t k = 0UZ; k < energy.size(); ++k) {
            spread += std::format("{}{:.4f}", k == 0UZ ? "" : ", ", energy[k]);
        }
        std::print("[measured] fs/16 through a 2x2 bank lands in port {} (energies: {})\n", best, spread);
    };

    "the device path gives the same channel split as the host"_test = [] {
        // where a SYCL backend exists this runs through processBulk(ctx, ...); where none does it falls back and
        // the comparison still has to hold. The two must agree whichever happened.
        std::vector<C> tone(16UZ);
        for (std::size_t n = 0UZ; n < tone.size(); ++n) {
            const float angle = 2.f * std::numbers::pi_v<float> * static_cast<float>(n) / 16.f;
            tone[n]           = C{std::cos(angle), std::sin(angle)};
        }
        const auto onHost   = runTone(tone, 2U, 2U, "host");
        const auto onDevice = runTone(tone, 2U, 2U, "host:sycl");

        expect(eq(onHost.size(), onDevice.size())) << "the two domains produced different channel counts";
        for (std::size_t k = 0UZ; k < onHost.size() && k < onDevice.size(); ++k) {
            const float tolerance = 0.05f * std::max(onHost[k], 1e-3f);
            expect(approx(onDevice[k], onHost[k], tolerance)) << std::format("channel {}: host {:.5f} against device {:.5f}", k, onHost[k], onDevice[k]);
        }
    };

    "a tone on a stage-1 boundary splits, by construction"_test = [] {
        // fs/4 is the stage-1 channel edge of a 2x2 bank: two ports share it evenly. This is a property of
        // cascading critically sampled stages, NOT a defect -- pinned so a future change cannot hide it.
        const std::vector<C> tone{C{1.f, 0.f}, C{0.f, 1.f}, C{-1.f, 0.f}, C{0.f, -1.f}};
        const auto           energy = runTone(tone, 2U, 2U);

        std::vector<float> sorted(energy);
        std::ranges::sort(sorted, std::greater<>{});
        expect(gt(sorted[0], 0.f)) << "no port carried anything";
        expect(approx(sorted[1], sorted[0], 0.15f * sorted[0])) << std::format("the two strongest ports hold {:.4f} and {:.4f}: a boundary tone should divide evenly", sorted[0], sorted[1]);
    };
    return 0;
}
