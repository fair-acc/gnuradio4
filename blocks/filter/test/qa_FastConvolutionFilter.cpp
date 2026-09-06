#include <boost/ut.hpp>

#include <cmath>
#include <format>
#include <numbers>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/filter/FastConvolution.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
/// run a repeating pattern through the block and return what the sink logged
[[nodiscard]] std::vector<float> run(const std::vector<float>& pattern, gr::property_map settings) {
    using namespace std::string_literals;
    using namespace gr::testing;

    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source = flow.emplaceBlock<TagSource<float>>({{"n_samples_max", gr::Size_t(8192)}, {"values", pattern}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<gr::filter::FastConvolutionFilter<float>>(std::move(settings));
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    boost::ut::expect(flow.connect(source, "out"s, dut, "in"s).has_value());
    boost::ut::expect(flow.connect(dut, "out"s, sink, "in"s).has_value());

    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());
    return {sink._samples.begin(), sink._samples.end()};
}

/// the answer the block exists to produce faster, evaluated the obvious way
[[nodiscard]] std::vector<float> convolveDirectly(std::span<const float> signal, std::span<const float> taps) {
    std::vector<float> out(signal.size(), 0.f);
    for (std::size_t n = 0UZ; n < signal.size(); ++n) {
        float acc = 0.f;
        for (std::size_t k = 0UZ; k < taps.size() && k <= n; ++k) {
            acc += taps[k] * signal[n - k];
        }
        out[n] = acc;
    }
    return out;
}
} // namespace

const boost::ut::suite<"FastConvolutionFilter"> _fastConvolution = [] {
    using namespace boost::ut;

    "a transform-domain filter is the filter it replaces"_test = [] {
        // the block is worth having only if it agrees with a tap-per-sample convolution, so that is what it is
        // held to -- over a chirp, which puts energy everywhere the filter has a response
        std::vector<float> taps(48);
        for (std::size_t k = 0UZ; k < taps.size(); ++k) { // a windowed lowpass, so the response is not flat
            const double centred = static_cast<double>(k) - 0.5 * static_cast<double>(taps.size() - 1UZ);
            const double sinc    = centred == 0.0 ? 0.25 : std::sin(0.25 * std::numbers::pi * centred) / (std::numbers::pi * centred);
            taps[k]              = static_cast<float>(sinc * (0.54 - 0.46 * std::cos(2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(taps.size() - 1UZ))));
        }

        std::vector<float> pattern(1024);
        for (std::size_t n = 0UZ; n < pattern.size(); ++n) {
            pattern[n] = static_cast<float>(std::sin(0.0005 * static_cast<double>(n) * static_cast<double>(n)));
        }

        const std::vector<float> got = run(pattern, {{"taps", std::vector<float>(taps.begin(), taps.end())}, {"outputs_per_frame", gr::Size_t(256)}});
        expect(gt(got.size(), 512UZ)) << "the block produced almost nothing";

        // the block runs the pattern repeatedly, so compare against a directly convolved run of the same length
        std::vector<float> repeated(got.size() + taps.size());
        for (std::size_t n = 0UZ; n < repeated.size(); ++n) {
            repeated[n] = pattern[n % pattern.size()];
        }
        const std::vector<float> reference = convolveDirectly(repeated, taps);

        // the first frame's outputs precede a full history, so start past one filter length
        // overlap-save discards the first `nTaps - 1` samples of every frame, so the stream it publishes sits that
        // far ahead of a convolution that starts from a cold history
        const std::size_t lag   = taps.size() - 1UZ;
        double            worst = 0.0;
        for (std::size_t n = taps.size(); n + lag < std::min(got.size(), reference.size()); ++n) {
            worst = std::max(worst, std::abs(static_cast<double>(got[n]) - static_cast<double>(reference[n + lag])));
        }
        expect(lt(worst, 1e-3)) << std::format("the transform disagrees with a direct convolution by {:.3e}", worst);
    };

    "a frame the real transform cannot factor still filters"_test = [] {
        // `outputs_per_frame` drives the frame size, and not every frame the block can be asked for is one the
        // real transform can take; the complex route exists for those and must give the same answer
        const std::vector<float> taps{0.2f, 0.3f, 0.3f, 0.2f};
        std::vector<float>       pattern(512);
        for (std::size_t n = 0UZ; n < pattern.size(); ++n) {
            pattern[n] = static_cast<float>(std::sin(0.03 * static_cast<double>(n)));
        }
        for (const gr::Size_t perFrame : {gr::Size_t(13), gr::Size_t(64), gr::Size_t(100)}) {
            const std::vector<float> got = run(pattern, {{"taps", taps}, {"outputs_per_frame", perFrame}});
            expect(gt(got.size(), 128UZ)) << std::format("outputs_per_frame {} produced almost nothing", perFrame);

            std::vector<float> repeated(got.size() + taps.size());
            for (std::size_t n = 0UZ; n < repeated.size(); ++n) {
                repeated[n] = pattern[n % pattern.size()];
            }
            const std::vector<float> reference = convolveDirectly(repeated, taps);
            const std::size_t        lag       = taps.size() - 1UZ;
            double                   worst     = 0.0;
            for (std::size_t n = taps.size(); n + lag < std::min(got.size(), reference.size()); ++n) {
                worst = std::max(worst, std::abs(static_cast<double>(got[n]) - static_cast<double>(reference[n + lag])));
            }
            expect(lt(worst, 1e-3)) << std::format("outputs_per_frame {} disagrees with a direct convolution by {:.3e}", perFrame, worst);
        }
    };
};

int main() { /* tests run from the suite */ }
