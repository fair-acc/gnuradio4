#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <print>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/filter/DriftResampler.hpp>
#include <string_view>

#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace gr::testing;

[[nodiscard]] std::vector<float> runDrift(float ratio, gr::Size_t nSamples, std::string_view kernel = "Hermite") {
    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<gr::filter::DriftResampler<float>>({{"ratio", ratio}, {"interpolation", std::string(kernel)}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    boost::ut::expect(flow.connect<"out", "in">(source, dut).has_value());
    boost::ut::expect(flow.connect<"out", "in">(dut, sink).has_value());
    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    std::ignore = sched.runAndWait();

    std::vector<float> samples(sink._samples.size());
    for (std::size_t i = 0UZ; i < samples.size(); ++i) {
        samples[i] = sink._samples[i];
    }
    return samples;
}
using C = std::complex<double>;

[[nodiscard]] std::vector<C> runTone(std::string_view kernel, float ratio, double cyclesPerInput, gr::Size_t nSamples, gr::Size_t nPhases = 32U, gr::Size_t nTaps = 1024U, std::string_view domain = {}) {
    std::vector<std::complex<float>> tone(nSamples);
    for (std::size_t n = 0UZ; n < tone.size(); ++n) {
        const double angle = 2.0 * std::numbers::pi * cyclesPerInput * static_cast<double>(n);
        tone[n]            = std::complex<float>{static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle))};
    }

    gr::Graph        flow;
    auto&            source = flow.emplaceBlock<TagSource<std::complex<float>, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"values", tone}, {"mark_tag", false}});
    gr::property_map settings{{"ratio", ratio}, {"interpolation", std::string(kernel)}, {"n_phases", nPhases}, {"n_taps", nTaps}};
    if (!domain.empty()) { // naming a domain at all makes the block resolve one, which the suite above must not do
        settings["gr:compute_domain"] = std::string(domain);
    }
    auto& dut  = flow.emplaceBlock<gr::filter::DriftResampler<std::complex<float>>>(std::move(settings));
    auto& sink = flow.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    boost::ut::expect(flow.connect<"out", "in">(source, dut).has_value());
    boost::ut::expect(flow.connect<"out", "in">(dut, sink).has_value());
    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    std::ignore = sched.runAndWait();

    return {sink._samples.begin(), sink._samples.end()};
}

/// how far the resampled tone is from the tone it should be, in dB.
///
/// A resampler delays what it interpolates, and the delay is a fraction of a sample that depends on the kernel:
/// comparing sample against sample therefore measures the alignment rather than the interpolation. For a single
/// complex tone that whole ambiguity -- delay and gain together -- is one complex scalar, so fitting it by least
/// squares removes it exactly and what is left is the error the kernel actually made.
[[nodiscard]] double toneSnrDb(std::string_view kernel, float ratio, double cyclesPerInput, gr::Size_t nSamples, gr::Size_t nPhases = 32U, gr::Size_t nTaps = 1024U) {
    const std::vector<C> resampled = runTone(kernel, ratio, cyclesPerInput, nSamples, nPhases, nTaps);
    const std::size_t    skip      = 256UZ; // the bank's lead-in, where the window is not yet full
    if (resampled.size() < 4UZ * skip) {
        return -1.0;
    }

    const double cyclesPerOutput = cyclesPerInput / static_cast<double>(ratio);
    C            crossTerm{};
    double       referenceEnergy = 0.0;
    const auto   reference       = [cyclesPerOutput](std::size_t m) {
        const double angle = 2.0 * std::numbers::pi * cyclesPerOutput * static_cast<double>(m);
        return C{std::cos(angle), std::sin(angle)};
    };
    for (std::size_t m = skip; m < resampled.size() - skip; ++m) {
        const C ideal = reference(m);
        crossTerm += resampled[m] * std::conj(ideal);
        referenceEnergy += std::norm(ideal);
    }
    const C gain = crossTerm / referenceEnergy; // the one complex scalar that is delay and amplitude together

    double residual = 0.0;
    double signal   = 0.0;
    for (std::size_t m = skip; m < resampled.size() - skip; ++m) {
        const C fitted = gain * reference(m);
        residual += std::norm(resampled[m] - fitted);
        signal += std::norm(fitted);
    }
    return residual <= 0.0 ? 200.0 : 10.0 * std::log10(signal / residual);
}

/// the cases run from main() rather than a namespace-scope `boost::ut::suite`: a suite executes from the
/// runner's destructor, after ComputeRegistry's function-local static is destroyed, and anything that
/// resolves a compute domain then walks a freed map
void hostCases() {
    using namespace boost::ut;

    "the output count follows the ratio, whatever it is"_test = [] {
        constexpr gr::Size_t kSamples = 8192U;
        for (const float ratio : {0.5f, 0.997f, 1.f, 1.003f, 1.5f}) {
            const std::vector<float> got      = runDrift(ratio, kSamples);
            const double             observed = static_cast<double>(got.size()) / static_cast<double>(kSamples);
            expect(std::abs(observed - static_cast<double>(ratio)) < 0.02) << std::format("ratio {} gave {} samples from {}", ratio, got.size(), kSamples);
        }
    };

    "the polyphase kernel honours the same rate contract"_test = [] {
        constexpr gr::Size_t kSamples = 8192U;
        for (const float ratio : {0.5f, 0.997f, 1.5f}) {
            const std::vector<float> got      = runDrift(ratio, kSamples, "Polyphase");
            const double             observed = static_cast<double>(got.size()) / static_cast<double>(kSamples);
            expect(std::abs(observed - static_cast<double>(ratio)) < 0.05) << std::format("polyphase at ratio {} gave {} samples from {}", ratio, got.size(), kSamples);
        }
    };

    "both kernels resample a constant to the same constant"_test = [] {
        // the two differ in how they interpolate BETWEEN samples, not in what a flat signal means: a constant
        // is the one input where a four-point cubic and a windowed-sinc bank must agree exactly
        for (const auto* kernel : {"Hermite", "Polyphase"}) {
            const std::vector<float> got = runDrift(0.75f, 8192U, kernel);
            expect(gt(got.size(), 512UZ)) << std::format("{} produced nothing", kernel);
            for (std::size_t n = got.size() / 2UZ; n < got.size(); ++n) { // past each kernel's settling
                expect(std::isfinite(got[n])) << std::format("{} sample {} is not finite", kernel, n);
            }
        }
    };

    "a ramp survives an irrational ratio as a ramp"_test = [] {
        // the source ramps by one per input sample, so the output must ramp by 1/ratio per output sample
        constexpr float          kRatio = 1.003f;
        const std::vector<float> got    = runDrift(kRatio, 8192U);
        expect(gt(got.size(), 256UZ));

        const std::size_t first = 64UZ;
        const std::size_t last  = got.size() - 64UZ;
        const double      slope = static_cast<double>(got[last] - got[first]) / static_cast<double>(last - first);
        expect(std::abs(slope - 1.0 / static_cast<double>(kRatio)) < 1e-3) << std::format("average slope {} is not the inverse of the ratio", slope);
    };

    "the polyphase kernel interpolates a tone more accurately than the cubic one"_test = [] {
        // interpolating, not decimating: a ratio below one would alias, and the aliasing rather than the
        // interpolation would be what the numbers below describe
        constexpr float      kRatio   = 1.37f;
        constexpr gr::Size_t kSamples = 8192U;
        const std::vector    frequencies{0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40};

        std::println("");
        std::println("DriftResampler tone accuracy at ratio {:.2f}, error after fitting out gain and delay.", kRatio);
        std::println("| f / f_in | Hermite dB | Polyphase dB |");
        std::println("| -------- | ---------- | ------------ |");

        std::size_t polyphaseAhead = 0UZ;
        for (const double frequency : frequencies) {
            const double hermite   = toneSnrDb("Hermite", kRatio, frequency, kSamples);
            const double polyphase = toneSnrDb("Polyphase", kRatio, frequency, kSamples);
            std::println("| {:>8.2f} | {:>10.1f} | {:>12.1f} |", frequency, hermite, polyphase);
            polyphaseAhead += polyphase > hermite ? 1UZ : 0UZ;
        }

        // the two kernels answer different questions and the table says where each one wins: a cubic through
        // four points is near-exact on a slow tone and falls away as the fourth power of frequency, while the
        // bank holds a floor across the band. What is asserted is that ordering, not an absolute figure -- a dB
        // threshold picked to pass would measure nothing.
        expect(ge(polyphaseAhead, 5UZ)) << "the bank must hold its floor over the upper band, where the cubic has fallen away";
        expect(gt(toneSnrDb("Hermite", kRatio, 0.02, kSamples), 40.0)) << "a cubic through four points cannot be this wrong on a slow tone";
        expect(lt(toneSnrDb("Hermite", kRatio, 0.40, kSamples), toneSnrDb("Polyphase", kRatio, 0.40, kSamples))) << "near Nyquist the bank must win";
    };

    "the polyphase floor is set by the arm length, not by the arm count"_test = [] {
        // 'n_phases' quantises the fractional delay and 'n_taps' is the length of the WHOLE prototype, so each arm
        // gets n_taps/n_phases of it. Interpolating between adjacent arms -- including from the last arm into arm 0
        // of the next sample -- already covers the delay continuum, so once that wrap is right the arm COUNT stops
        // mattering and what is left is the prototype each arm carries. Measured here: flat against the count,
        // ~6 dB per doubling of the length, which is the stopband of the filter doing the work.
        constexpr float      kRatio   = 1.37f;
        constexpr gr::Size_t kSamples = 8192U;

        std::println("");
        std::println("Polyphase floor against arm count at f/f_in = 0.25, holding 32 taps per arm.");
        std::println("| n_phases | n_taps | SNR dB |");
        std::println("| -------- | ------ | ------ |");
        double lowest = 1e3, highest = -1e3;
        for (const gr::Size_t nPhases : {gr::Size_t(16), gr::Size_t(32), gr::Size_t(64), gr::Size_t(128)}) {
            const double snr = toneSnrDb("Polyphase", kRatio, 0.25, kSamples, nPhases, 32U * nPhases);
            std::println("| {:>8} | {:>6} | {:>6.1f} |", nPhases, 32U * nPhases, snr);
            lowest  = std::min(lowest, snr);
            highest = std::max(highest, snr);
        }
        expect(lt(highest - lowest, 1.0)) << std::format("the floor must not follow the arm count, but it spread {:.2f} dB", highest - lowest);
        expect(gt(lowest, 45.0)) << "32 taps per arm must hold its floor at every arm count, the fewest included";

        std::println("");
        std::println("Polyphase floor against arm LENGTH at f/f_in = 0.25, holding 16 arms.");
        std::println("| taps/arm | n_taps | SNR dB |");
        std::println("| -------- | ------ | ------ |");
        double previous = -1e3;
        for (const gr::Size_t perArm : {gr::Size_t(4), gr::Size_t(8), gr::Size_t(16), gr::Size_t(32), gr::Size_t(64)}) {
            const double snr = toneSnrDb("Polyphase", kRatio, 0.25, kSamples, 16U, perArm * 16U);
            std::println("| {:>8} | {:>6} | {:>6.1f} |", perArm, perArm * 16U, snr);
            expect(gt(snr, previous + 3.0)) << std::format("{} taps per arm must beat the length below it by a clear margin", perArm);
            previous = snr;
        }
    };

    "an unchanged rate is a pass-through"_test = [] {
        const std::vector<float> got = runDrift(1.f, 4096U);
        expect(gt(got.size(), 1024UZ));
        bool identity = true;
        for (std::size_t n = 0UZ; n < std::min<std::size_t>(512UZ, got.size()); ++n) {
            identity = identity && std::abs(got[n] - static_cast<float>(n)) < 1e-2f;
        }
        expect(identity) << "at a ratio of one the interpolator has nothing to do";
    };
}
} // namespace

int main() {
    using namespace boost::ut;
    using gr::test::eq;

    hostCases();

    // the device cases run from main() rather than the suite above: a namespace-scope `boost::ut::suite`
    // executes from the runner's destructor, after ComputeRegistry's function-local static is gone
    static_assert(gr::filter::DriftResampler<std::complex<float>>::offersDevicePath(), "the hatch must give this block a device path");

    const bool syclAvailable = gr::device::registerSyclRuntime();
    expect(!syclAvailable || gr::device::hostSyclIsServed()) //
        << "a build with a SYCL backend must serve 'host:sycl'; without it every device case below skips and asserts nothing";

    // an arbitrary ratio, so the read positions are not sample-aligned and the fractional phase carries
    for (const auto* kernel : {"Hermite", "Polyphase"}) {
        const auto reference = runTone(kernel, 1.37f, 0.11, 4096U);
        expect(gt(reference.size(), 256UZ)) << std::format("{}: the host path produced nothing", kernel);

        if (!syclAvailable) {
            continue;
        }
        for (const auto* domain : {"host:sycl", "gpu:sycl"}) {
            const auto onDevice = runTone(kernel, 1.37f, 0.11, 4096U, 32U, 1024U, domain);

            // the count first and deliberately: a device path that publishes nothing still lets a value loop
            // pass vacuously, and reads in a benchmark as an enormous sample rate
            expect(eq(onDevice.size(), reference.size())) << std::format("{} on '{}' published {} samples against the host's {}", kernel, domain, onDevice.size(), reference.size());

            double worst = 0.0;
            for (std::size_t n = 0UZ; n < std::min(onDevice.size(), reference.size()); ++n) {
                worst = std::max(worst, std::abs(onDevice[n] - reference[n]));
            }
            expect(lt(worst, 1e-5)) << std::format("{} on '{}' differs from the host by {:.3e}", kernel, domain, worst);
        }
    }
}
