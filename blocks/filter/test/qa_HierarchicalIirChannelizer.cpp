#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <print>
#include <string>
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

struct RunResult {
    std::vector<float> energy; // mean |sample| per channel port, past settling
    float              inputRms = 0.f;
};

[[nodiscard]] RunResult runTone(std::size_t periodSamples, gr::Size_t nChannels, gr::Size_t nSections, std::string_view domain = "host") {
    using namespace std::string_literals;
    using namespace gr::testing;

    std::vector<C> tone(periodSamples);
    for (std::size_t n = 0UZ; n < periodSamples; ++n) {
        const float angle = 2.f * std::numbers::pi_v<float> * static_cast<float>(n) / static_cast<float>(periodSamples);
        tone[n]           = C{std::cos(angle), std::sin(angle)};
    }

    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source = flow.emplaceBlock<TagSource<C>>({{"n_samples_max", gr::Size_t(32768)}, {"values", tone}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<gr::filter::HierarchicalIirChannelizer<C>>({{"n_channels", nChannels}, {"n_sections", nSections}, {"outputs_per_frame", gr::Size_t(64)}, {"gr:compute_domain", std::string(domain)}});

    boost::ut::expect(flow.connect(source, "out"s, dut, "in"s).has_value());

    std::vector<TagSink<C, ProcessFunction::USE_PROCESS_ONE>*> sinks;
    for (gr::Size_t k = 0U; k < nChannels; ++k) {
        sinks.push_back(std::addressof(flow.emplaceBlock<TagSink<C, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}})));
        boost::ut::expect(flow.connect(dut, "out#"s + std::to_string(k), *sinks[k], "in"s).has_value());
    }

    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());

    RunResult result;
    result.energy.resize(nChannels, 0.f);
    for (std::size_t k = 0UZ; k < nChannels; ++k) {
        const auto&       samples = sinks[k]->_samples;
        const std::size_t from    = samples.size() / 2UZ;
        for (std::size_t i = from; i < samples.size(); ++i) {
            result.energy[k] += std::abs(samples[i]);
        }
        result.energy[k] /= static_cast<float>(std::max(std::size_t{1}, samples.size() - from));
    }
    result.inputRms = 1.f; // the tone is unit magnitude by construction
    return result;
}
/// the cases run from main() rather than a namespace-scope `boost::ut::suite`: a suite executes from the
/// runner's destructor, after ComputeRegistry's function-local static is gone, and a case that resolves a
/// compute domain then walks a freed map
void hostCases() {
    using namespace boost::ut;
    using gr::test::eq;

    "the tree only reaches powers of two"_test = [] {
        gr::filter::HierarchicalIirChannelizer<C> eight({{"n_channels", gr::Size_t(8)}});
        eight.settings().init();
        std::ignore = eight.settings().applyStagedParameters();
        expect(eq(eight.out.size(), 8UZ)) << "8 is a power of two";
        expect(eq(eight.depth(), 3UZ)) << "three halvings reach eight bands";

        // a half-band tree cannot express 12 channels, so it rounds DOWN rather than silently mis-splitting
        gr::filter::HierarchicalIirChannelizer<C> twelve({{"n_channels", gr::Size_t(12)}});
        twelve.settings().init();
        std::ignore = twelve.settings().applyStagedParameters();
        expect(eq(twelve.out.size(), 8UZ)) << "12 rounds down to 8";
    };

    "each tree node gets its own state"_test = [] {
        gr::filter::HierarchicalIirChannelizer<C> block({{"n_channels", gr::Size_t(8)}, {"n_sections", gr::Size_t(3)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();

        // a binary tree over 8 leaves has 7 internal nodes, and each needs both branches' state
        expect(eq(block._state.size(), 7UZ * block.stateStride())) << std::format("state holds {} values for stride {}", block._state.size(), block.stateStride());
        expect(gt(block.alphas_even.size() + block.alphas_odd.size(), 0UZ)) << "no allpass coefficients were designed";
    };

    "a tone concentrates in one channel"_test = [] {
        // period 32 over 8 channels: well inside one leaf rather than on a crossover
        const auto run = runTone(32UZ, 8U, 3U);

        std::size_t best  = 0UZ;
        float       bestE = -1.f;
        float       total = 0.f;
        for (std::size_t k = 0UZ; k < run.energy.size(); ++k) {
            total += run.energy[k];
            if (run.energy[k] > bestE) {
                bestE = run.energy[k];
                best  = k;
            }
        }
        expect(gt(bestE, 0.f)) << "no channel carried anything";
        expect(gt(bestE, 0.5f * total)) << std::format("the strongest channel (#{}) holds {:.4f} of {:.4f}", best, bestE, total);

        std::string spread;
        for (std::size_t k = 0UZ; k < run.energy.size(); ++k) {
            spread += std::format("{}{:.4f}", k == 0UZ ? "" : ", ", run.energy[k]);
        }
        std::print("[measured] IIR tree, period-32 tone, 8 channels -> port {} (energies: {})\n", best, spread);
    };

    "more sections sharpen the split"_test = [] {
        const auto few  = runTone(32UZ, 8U, 1U);
        const auto many = runTone(32UZ, 8U, 4U);

        const auto concentration = [](const std::vector<float>& e) {
            float best = 0.f, total = 0.f;
            for (const float v : e) {
                total += v;
                best = std::max(best, v);
            }
            return total > 0.f ? best / total : 0.f;
        };
        expect(ge(concentration(many.energy), concentration(few.energy) - 0.02f)) //
            << std::format("4 sections concentrate {:.3f} against 1 section's {:.3f}", concentration(many.energy), concentration(few.energy));
    };
}

} // namespace

int main() {
    using namespace boost::ut;
    using gr::test::eq;

    hostCases();

    const bool syclAvailable = gr::device::registerSyclRuntime();
    expect(!syclAvailable || gr::device::hostSyclIsServed()) //
        << "a build with a SYCL backend must serve 'host:sycl'; without it every device case below skips and asserts nothing";
    if (!syclAvailable) {
        return 0;
    }

    // the tree dispatches on a device (it has a hatch) but its parallelism is only across bands, so this
    // asserts that the ANSWER is the same there, not that it is faster -- it is not, and deliberately so
    for (const gr::Size_t channels : {4U, 8U}) {
        const auto reference = runTone(16UZ, channels, 3U, "host");
        for (const auto* domain : {"host:sycl", "gpu:sycl"}) {
            const auto onDevice = runTone(16UZ, channels, 3U, domain);
            expect(eq(onDevice.energy.size(), reference.energy.size())) << std::format("{} channels on '{}': {} ports against the host's {}", channels, domain, onDevice.energy.size(), reference.energy.size());
            for (std::size_t k = 0UZ; k < std::min(onDevice.energy.size(), reference.energy.size()); ++k) {
                expect(approx(onDevice.energy[k], reference.energy[k], 1e-3f)) << std::format("{} channels on '{}': channel {} reads {:.6f} against the host's {:.6f}", channels, domain, k, onDevice.energy[k], reference.energy[k]);
            }
        }
    }
}
