#include <boost/ut.hpp>

#include <bit>
#include <numbers>

#include <complex>
#include <format>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/filter/DriftResampler.hpp>
#include <gnuradio-4.0/filter/FrequencyXlatingFilter.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using C = std::complex<float>;

/// run a repeating pattern through the block and return what the sink logged
[[nodiscard]] std::vector<C> run(const std::vector<C>& pattern, gr::property_map settings) {
    using namespace std::string_literals;
    using namespace gr::testing;

    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source = flow.emplaceBlock<TagSource<C>>({{"n_samples_max", gr::Size_t(8192)}, {"values", pattern}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<gr::filter::FrequencyXlatingFilter<C>>(std::move(settings));
    auto&     sink   = flow.emplaceBlock<TagSink<C, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    boost::ut::expect(flow.connect(source, "out"s, dut, "in"s).has_value());
    boost::ut::expect(flow.connect(dut, "out"s, sink, "in"s).has_value());

    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());
    return {sink._samples.begin(), sink._samples.end()};
}
/// a device block feeding one that has no device path at all: the edge between them is sized before either
/// block has decided anything, so if it is sized for the device the resampler is handed device-only memory to
/// read on the host. Returns what the sink logged, which on a wrong edge is a fault rather than a number.
[[nodiscard]] std::vector<C> runThroughFallbackConsumer(const std::vector<C>& pattern, std::string_view domain) {
    using namespace std::string_literals;
    using namespace gr::testing;

    gr::property_map filterSettings{{"sample_rate", 1'000.f}, {"frequency", 250.0}, {"cutoff", 200.f}, {"filter_order", gr::Size_t(4)}, {"decimation", gr::Size_t(2)}, {"samples_per_frame", gr::Size_t(64)}};
    filterSettings["gr:compute_domain"] = std::string(domain);

    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source    = flow.emplaceBlock<TagSource<C>>({{"n_samples_max", gr::Size_t(4096)}, {"values", pattern}, {"mark_tag", false}});
    auto&     xlating   = flow.emplaceBlock<gr::filter::FrequencyXlatingFilter<C>>(std::move(filterSettings));
    auto&     resampler = flow.emplaceBlock<gr::filter::DriftResampler<C>>({{"ratio", 1.f}});
    auto&     sink      = flow.emplaceBlock<TagSink<C, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    boost::ut::expect(flow.connect(source, "out"s, xlating, "in"s).has_value());
    boost::ut::expect(flow.connect(xlating, "out"s, resampler, "in"s).has_value());
    boost::ut::expect(flow.connect(resampler, "out"s, sink, "in"s).has_value());

    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());
    return {sink._samples.begin(), sink._samples.end()};
}

/// the cases run from main() rather than a namespace-scope `boost::ut::suite`: a suite executes from the
/// runner's destructor, after ComputeRegistry's function-local static is gone, and any case that resolves a
/// compute domain then walks a freed map
void hostCases() {
    using namespace boost::ut;
    using gr::test::eq;

    "the walked mixer and the indexed one are the same translation"_test = [] {
        // the host rotates a phasor while a work item computes its own angle, so the two must agree to the same
        // tolerance the device equivalence cases use. What is really watched here is drift: a recurrence
        // accumulates error, and this pins how far it can get before anyone notices.
        for (const std::size_t n : {256UZ, 4096UZ, 65536UZ}) {
            std::vector<C> pattern(n);
            for (std::size_t i = 0UZ; i < n; ++i) {
                pattern[i] = C{std::sin(0.01f * static_cast<float>(i)), std::cos(0.013f * static_cast<float>(i))};
            }
            std::vector<float> walkedReal(n), walkedImag(n), indexedReal(n), indexedImag(n);
            const double       step = -2.0 * std::numbers::pi * 0.137; // an awkward step, so nothing repeats early
            const double       base = 0.3;

            using Filter = gr::filter::FrequencyXlatingFilter<C>;
            Filter::mixWindow(pattern.data(), walkedReal.data(), walkedImag.data(), n, base, step);
            for (std::size_t i = 0UZ; i < n; ++i) {
                Filter::mixInto(pattern.data(), indexedReal.data(), indexedImag.data(), i, base, step);
            }

            double worst = 0.0;
            for (std::size_t i = 0UZ; i < n; ++i) {
                worst = std::max(worst, static_cast<double>(std::abs(walkedReal[i] - indexedReal[i])));
                worst = std::max(worst, static_cast<double>(std::abs(walkedImag[i] - indexedImag[i])));
            }
            expect(lt(worst, 1e-4)) << std::format("over {} samples the phasor drifted {:.3e} from the closed form", n, worst);
        }
    };
    "the rate contract follows decimation"_test = [] {
        using namespace std::string_literals;
        gr::filter::FrequencyXlatingFilter<C> block({{"filter_domain", "Time"s}, {"decimation", gr::Size_t(4)}, {"samples_per_frame", gr::Size_t(64)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();
        expect(eq(static_cast<std::size_t>(block.output_chunk_size), 64UZ));
        expect(eq(static_cast<std::size_t>(block.stride), 256UZ)) << "64 outputs at 4:1";
        // an FIR window carries its own lead-in and consecutive windows overlap by it, so the presented chunk
        // is longer than the consumed one -- that overlap is what makes the FIR path stateless
        expect(eq(static_cast<std::size_t>(block.input_chunk_size), 256UZ + block._nCoefficients - 1UZ)) << "the window carries taps-1 samples of lead-in";
        expect(gt(block._nCoefficients, 256UZ)) << "order 32 asks for a narrow transition, so the Kaiser estimate is long";
    };

    "an IIR design consumes exactly what it is shown"_test = [] {
        using namespace std::string_literals;
        // a recursion's state depends on every input it was given, so overlapping windows would feed it the
        // same samples twice; only the FIR path may carry a lead-in
        gr::filter::FrequencyXlatingFilter<C> block({{"filter_type", "IIR"s}, {"filter_order", gr::Size_t(4)}, {"decimation", gr::Size_t(4)}, {"samples_per_frame", gr::Size_t(64)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();
        expect(eq(static_cast<std::size_t>(block.input_chunk_size), static_cast<std::size_t>(block.stride))) << "an IIR window must not overlap";
        expect(eq(static_cast<std::size_t>(block.input_chunk_size), 256UZ));
    };

    "the window lead-in carries, so a split run matches a whole one"_test = [] {
        // the FIR path is stateless only because the framework re-presents taps-1 samples of lead-in. If that
        // overlap were wrong the output would still look plausible, with a transient at every frame boundary --
        // so compare one stream filtered whole against the same stream filtered in two halves.
        std::vector<C> tone(4096UZ);
        for (std::size_t n = 0UZ; n < tone.size(); ++n) {
            const float a = 2.f * std::numbers::pi_v<float> * 0.11f * static_cast<float>(n);
            tone[n]       = C{std::cos(a), std::sin(a)};
        }
        const gr::property_map settings{{"sample_rate", 1'000.f}, {"frequency", 110.0}, {"cutoff", 200.f}, {"filter_order", gr::Size_t(4)}, {"decimation", gr::Size_t(2)}, {"samples_per_frame", gr::Size_t(64)}};

        const auto whole = run(tone, settings);
        expect(gt(whole.size(), 64UZ)) << "nothing came through";

        // the same block, driven with a frame size that makes the seam fall in a different place
        gr::property_map other     = settings;
        other["samples_per_frame"] = gr::Size_t(32);
        const auto split           = run(tone, other);

        const std::size_t common = std::min(whole.size(), split.size());
        expect(gt(common, 64UZ)) << "too few samples to compare";
        for (std::size_t n = 0UZ; n < common; ++n) {
            expect(approx(std::abs(whole[n] - split[n]), 0.f, 1e-4f)) << std::format("sample {} differs: {} vs {}", n, std::abs(whole[n]), std::abs(split[n]));
        }
    };

    "the transform domain declares a power-of-two frame and the overlap it discards"_test = [] {
        using namespace std::string_literals;
        gr::filter::FrequencyXlatingFilter<C> block({{"filter_domain", "Frequency"s}, {"filter_order", gr::Size_t(4)}, {"decimation", gr::Size_t(2)}, {"samples_per_frame", gr::Size_t(64)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();

        const std::size_t frameSize = static_cast<std::size_t>(block.input_chunk_size);
        expect(eq(frameSize, std::bit_ceil(frameSize))) << "overlap-save transforms a power-of-two frame";
        expect(eq(frameSize, static_cast<std::size_t>(block.stride) + block._nLead)) << "the frame is the step plus the lead-in it re-presents";
        expect(ge(block._nLead, block._nCoefficients - 1UZ)) << "the discarded wrap-around must cover the filter";
        expect(eq(static_cast<std::size_t>(block.stride) % 2UZ, 0UZ)) << "the step must be whole decimated outputs, or the phase moves between frames";
    };

    "a device domain keeps the tap form, whatever the tap count asks for"_test = [] {
        using namespace std::string_literals;
        // the tap spectrum is a host vector: a kernel reaching it is valid on 'host:sycl' and a fault on a real
        // device, so 'Auto' must not choose the transform there however long the filter is
        gr::filter::FrequencyXlatingFilter<C> block({{"gr:compute_domain", "gpu:sycl"s}, {"filter_order", gr::Size_t(32)}, {"samples_per_frame", gr::Size_t(64)}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();
        expect(gt(block._nCoefficients, 128UZ)) << "the design must be past the threshold for this case to mean anything";
        expect(eq(static_cast<std::size_t>(block.input_chunk_size), 64UZ + block._nCoefficients - 1UZ)) << "a device domain declares the tap form's window";
    };

    "the two domains are the same filter"_test = [] {
        using namespace std::string_literals;
        // the discard length is the one thing overlap-save gets silently wrong: too short or too long still
        // produces plausible output, with a transient at every frame boundary. The tap form is the reference.
        std::vector<C> chirp(4096UZ);
        for (std::size_t n = 0UZ; n < chirp.size(); ++n) {
            const float a = 2.f * std::numbers::pi_v<float> * (0.03f + 1e-5f * static_cast<float>(n)) * static_cast<float>(n);
            chirp[n]      = C{std::cos(a), std::sin(a)};
        }
        gr::property_map settings{{"sample_rate", 1'000.f}, {"frequency", 60.0}, {"cutoff", 150.f}, {"filter_order", gr::Size_t(4)}, {"decimation", gr::Size_t(2)}, {"samples_per_frame", gr::Size_t(64)}};

        settings["filter_domain"] = "Time"s;
        const auto tapForm        = run(chirp, settings);
        settings["filter_domain"] = "Frequency"s;
        const auto transformForm  = run(chirp, settings);

        const std::size_t common = std::min(tapForm.size(), transformForm.size());
        expect(gt(common, 256UZ)) << std::format("too few samples to compare: {} against {}", tapForm.size(), transformForm.size());
        float worst = 0.f;
        for (std::size_t n = 0UZ; n < common; ++n) {
            worst = std::max(worst, std::abs(tapForm[n] - transformForm[n]));
        }
        expect(lt(worst, 1e-3f)) << std::format("the transform differs from the tap form by {:.3e}", worst);
    };

    "the transform seam carries, so a split run matches a whole one"_test = [] {
        using namespace std::string_literals;
        std::vector<C> tone(4096UZ);
        for (std::size_t n = 0UZ; n < tone.size(); ++n) {
            const float a = 2.f * std::numbers::pi_v<float> * 0.11f * static_cast<float>(n);
            tone[n]       = C{std::cos(a), std::sin(a)};
        }
        gr::property_map settings{{"filter_domain", "Frequency"s}, {"sample_rate", 1'000.f}, {"frequency", 110.0}, {"cutoff", 200.f}, {"filter_order", gr::Size_t(4)}, {"decimation", gr::Size_t(2)}, {"samples_per_frame", gr::Size_t(64)}};

        const auto whole = run(tone, settings);
        expect(gt(whole.size(), 64UZ)) << "nothing came through";

        // a different requested frame rounds to a different power of two, so the seams fall elsewhere
        gr::property_map other     = settings;
        other["samples_per_frame"] = gr::Size_t(200);
        const auto split           = run(tone, other);

        const std::size_t common = std::min(whole.size(), split.size());
        expect(gt(common, 64UZ)) << "too few samples to compare";
        float worst = 0.f;
        for (std::size_t n = 0UZ; n < common; ++n) {
            worst = std::max(worst, std::abs(whole[n] - split[n]));
        }
        expect(lt(worst, 1e-3f)) << std::format("the two frame sizes differ by {:.3e}", worst);
    };

    "a tone at the centre frequency comes out at DC"_test = [] {
        // at fs = 8 Hz, {1, i, -1, -i} repeats every 4 samples: a tone at +2 Hz
        const std::vector<C> tone{C{1.f, 0.f}, C{0.f, 1.f}, C{-1.f, 0.f}, C{0.f, -1.f}};
        const auto           samples = run(tone, {{"sample_rate", 8.f}, {"frequency", 2.0}, {"cutoff", 1.f}, {"filter_order", gr::Size_t(32)}, {"samples_per_frame", gr::Size_t(256)}});

        expect(gt(samples.size(), 512UZ)) << "nothing came through";
        const std::size_t from = samples.size() / 2UZ; // past the filter's settling
        float             mean = 0.f;
        for (std::size_t i = from; i < samples.size(); ++i) {
            mean += std::abs(samples[i]);
        }
        mean /= static_cast<float>(samples.size() - from);
        expect(gt(mean, 0.3f)) << std::format("the translated tone averages only {:.5f}", mean);

        // translated to DC, consecutive samples must barely differ -- that is what distinguishes DC from a tone
        for (std::size_t i = from + 1UZ; i < samples.size(); ++i) {
            expect(lt(std::abs(samples[i] - samples[i - 1UZ]), 0.1f * mean)) << std::format("sample {} moved by {:.5f} against level {:.5f}: not at DC", i, std::abs(samples[i] - samples[i - 1UZ]), mean);
        }
    };

    "a tone outside the passband is rejected"_test = [] {
        // the same +2 Hz tone, but translating by 0 leaves it at 2 Hz, well outside a 0.5 Hz cutoff
        const std::vector<C> tone{C{1.f, 0.f}, C{0.f, 1.f}, C{-1.f, 0.f}, C{0.f, -1.f}};
        const auto           samples = run(tone, {{"sample_rate", 8.f}, {"frequency", 0.0}, {"cutoff", 0.5f}, {"filter_order", gr::Size_t(64)}, {"samples_per_frame", gr::Size_t(256)}});

        expect(gt(samples.size(), 512UZ)) << "nothing came through";
        const std::size_t from = samples.size() / 2UZ;
        float             mean = 0.f;
        for (std::size_t i = from; i < samples.size(); ++i) {
            mean += std::abs(samples[i]);
        }
        mean /= static_cast<float>(samples.size() - from);
        expect(lt(mean, 0.2f)) << std::format("an out-of-band tone still averages {:.5f}", mean);
    };
    "a gr:frequency tag retunes the band mid-stream"_test = [] {
        using namespace std::string_literals;
        using namespace gr::testing;

        // the +2 Hz tone again. `frequency` is deliberately NOT named in the constructor: a setting given
        // there stops being auto-updated from tags, so the whole retune has to arrive as tags.
        const std::vector<C> tone{C{1.f, 0.f}, C{0.f, 1.f}, C{-1.f, 0.f}, C{0.f, -1.f}};

        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     source = flow.emplaceBlock<TagSource<C>>({{"n_samples_max", gr::Size_t(8192)}, {"values", tone}, {"mark_tag", false}});
        auto&     dut    = flow.emplaceBlock<gr::filter::FrequencyXlatingFilter<C>>({{"sample_rate", 8.f}, {"cutoff", 1.f}, {"filter_order", gr::Size_t(32)}, {"samples_per_frame", gr::Size_t(128)}});
        auto&     sink   = flow.emplaceBlock<TagSink<C, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

        source._tags.emplace_back(0UZ, gr::property_map{{"frequency", 0.0}});    // start off-tune: reject it
        source._tags.emplace_back(4096UZ, gr::property_map{{"frequency", 2.0}}); // retune onto the tone

        expect(flow.connect(source, "out"s, dut, "in"s).has_value());
        expect(flow.connect(dut, "out"s, sink, "in"s).has_value());

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());

        const auto& samples = sink._samples;
        expect(gt(samples.size(), 512UZ)) << "nothing came through";

        const std::size_t third = samples.size() / 3UZ;
        float             early = 0.f;
        float             late  = 0.f;
        for (std::size_t i = 0UZ; i < third; ++i) {
            early += std::abs(samples[i]);
        }
        for (std::size_t i = samples.size() - third; i < samples.size(); ++i) {
            late += std::abs(samples[i]);
        }
        early /= static_cast<float>(third);
        late /= static_cast<float>(third);

        expect(gt(late, 4.f * early)) << std::format("before the retune the band averaged {:.5f}, after it {:.5f}: the tag did not take effect", early, late);
    };
}

} // namespace

int main() {
    using namespace boost::ut;

    using gr::test::eq;

    hostCases();

    // the device cases must run inside main(): a namespace-scope `boost::ut::suite` executes from the runner's
    // destructor, after ComputeRegistry's function-local static has been destroyed, and resolving a backend
    // then walks a freed map
    const bool syclAvailable = gr::device::registerSyclRuntime();
    expect(!syclAvailable || gr::device::hostSyclIsServed()) //
        << "a build with a SYCL backend must serve 'host:sycl'; without it every device case below skips and asserts nothing";

    // a tone at +0.25 of the sample rate, which the NCO translates to DC
    std::vector<C> tone{C{1.f, 0.f}, C{0.f, 1.f}, C{-1.f, 0.f}, C{0.f, -1.f}};

    const auto atDomain = [&tone](std::string_view type, std::string_view domain) {
        gr::property_map settings{{"sample_rate", 1'000.f}, {"frequency", 250.0}, {"cutoff", 200.f}, {"filter_type", std::string(type)}, {"filter_order", gr::Size_t(4)}, {"decimation", gr::Size_t(2)}, {"samples_per_frame", gr::Size_t(64)}};
        settings["gr:compute_domain"] = std::string(domain);
        return run(tone, std::move(settings));
    };

    for (const auto* type : {"FIR", "IIR"}) {
        const auto reference = atDomain(type, "host");
        expect(gt(reference.size(), 64UZ)) << std::format("{}: the host path produced nothing", type);

        if (!syclAvailable) {
            continue;
        }
        for (const auto* domain : {"host:sycl", "gpu:sycl"}) {
            const auto onDevice = atDomain(type, domain);

            // the count is asserted first and deliberately: a device path that publishes nothing still lets a
            // magnitude loop pass vacuously, and reads in a benchmark as an enormous sample rate
            expect(eq(onDevice.size(), reference.size())) << std::format("{} on '{}' published {} samples against the host's {}", type, domain, onDevice.size(), reference.size());

            float worst = 0.f;
            for (std::size_t n = 0UZ; n < std::min(onDevice.size(), reference.size()); ++n) {
                worst = std::max(worst, std::abs(onDevice[n] - reference[n]));
            }
            expect(lt(worst, 1e-4f)) << std::format("{} on '{}' differs from the host by {:.3e}", type, domain, worst);
        }
    }

    // `DriftResampler` has no device path by design -- its output count per input is not fixed, so it owns its
    // own accounting and stays sequential. Putting it downstream of a device block is therefore the case where
    // an edge sized from a declared domain rather than from what the consumer can actually do goes wrong.
    const auto throughFallback = runThroughFallbackConsumer(tone, "host");
    expect(gt(throughFallback.size(), 64UZ)) << "the host reference produced nothing";
    if (syclAvailable) {
        for (const auto* domain : {"host:sycl", "gpu:sycl"}) {
            const auto onDevice = runThroughFallbackConsumer(tone, domain);
            expect(eq(onDevice.size(), throughFallback.size())) << std::format("a device producer feeding a host-only consumer on '{}' published {} samples against {}", domain, onDevice.size(), throughFallback.size());

            float worst = 0.f;
            for (std::size_t n = 0UZ; n < std::min(onDevice.size(), throughFallback.size()); ++n) {
                worst = std::max(worst, std::abs(onDevice[n] - throughFallback[n]));
            }
            expect(lt(worst, 1e-4f)) << std::format("'{}' into a host-only consumer differs from the host by {:.3e}", domain, worst);
        }
    }

    // the dispatch tier is what the device rows above actually exercise, and it is decided by a signature: a
    // change to constness or noexcept would move this block between tiers without failing any case here
    static_assert(gr::filter::FrequencyXlatingFilter<C>::offersDevicePath(), "the block must keep a device path");
}
