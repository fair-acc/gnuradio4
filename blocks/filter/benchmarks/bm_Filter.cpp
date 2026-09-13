#include <chrono>
#include <format>
#include <print>
#include <string_view>
#include <tuple>
#include <vector>

#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/filter/time_domain_filter.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include "device_test_helpers.hpp"

/*
 * At what length does a transform beat a tap, and what does an IIR cost instead? Nothing here asserts a number --
 * a throughput is a reading, not a contract, and a benchmark that fails a build on a number fails it on an
 * unrelated machine.
 *
 * The three answers to one low-pass are compared on the same signal: the direct FIR pays per tap, the overlap-save
 * FIR pays per frame regardless of tap count, and the IIR pays neither because it reaches the same corner
 * frequency in a handful of coefficients. The second table asks the separate question of which of them can leave
 * the host at all.
 */

using gr::test::servedDomains;

namespace {

constexpr gr::Size_t kIirOrder = 4U; // the order that replaces a long FIR, not a tap count
/// each length gets the window it deserves, as the canonical helper prescribes -- a fixed frame starves the
/// long filters, which then complete no window at all and report no measurement
[[nodiscard]] gr::Size_t frameOutputsFor(std::size_t nTaps) { return static_cast<gr::Size_t>(gr::test::windowForFilterLength(nTaps)); }

[[nodiscard]] std::vector<float> lowPassTaps(std::size_t nTaps) {
    std::vector<float> taps(nTaps, 1.0f / static_cast<float>(nTaps)); // a box car: the length is what is measured
    return taps;
}

/// input samples per second, so that a decimating or frame-based block is not flattered by counting its outputs
template<typename TBlock, typename TPrepare>
[[nodiscard]] double megaSamplesPerSecond(std::string_view domain, TPrepare prepare, gr::Size_t nSamples, int attempts) {
    using namespace gr::testing;

    struct Pass {
        std::size_t               consumed = 0UZ;
        std::size_t               produced = 0UZ;
        std::chrono::microseconds elapsed{0};
    };
    const auto once = [&](gr::Size_t runSamples) -> Pass {
        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        // neither end may be the thing measured. The default tag sink keeps every sample it sees, which costs more per
        // sample than most of the filters below and capped the whole table at ~130 MSample/s; a sink that only counts
        // lifts that to ~850. The source stays the bulk tag source because every other source in the tree fills its
        // span one sample at a time, which measured less than half this rate.
        auto& source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", runSamples}, {"mark_tag", false}});
        auto& dut    = flow.emplaceBlock<TBlock>({{"gr:compute_domain", std::string(domain)}});
        auto& sink   = flow.emplaceBlock<CountingSink<float>>({});

        prepare(dut);
        if constexpr (requires { dut.settingsChanged(gr::property_map{}, gr::property_map{}); }) {
            dut.settingsChanged({}, {}); // the coefficients 'prepare' just set have to be designed before the run
        }

        std::ignore = flow.connect<"out", "in">(source, dut);
        std::ignore = flow.connect<"out", "in">(dut, sink);

        gr::scheduler::Simple<> sched;
        if (!sched.exchange(std::move(flow)).has_value()) {
            return Pass{};
        }
        const auto started = std::chrono::steady_clock::now();
        gr::test::runAbsorbingRefusal(sched);
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started);
        // the INPUT rate is the comparable one -- a frame-based block emits fewer samples than it consumes, so
        // counting its outputs would make it look slower than a block doing the same work per input sample. The
        // numerator is what the source actually emitted, so a graph that stalls reads as no measurement at all
        return Pass{.consumed = static_cast<std::size_t>(source._nSamplesProduced), .produced = static_cast<std::size_t>(sink.count), .elapsed = elapsed};
    };

    // the warm-up also sizes the run. A cell's cost spans six orders of magnitude across this table -- a biquad on a
    // host against a thousand-tap cascade as a single work item on a GPU -- so a sample count chosen for the fast
    // cells would leave the slow ones running for minutes. Scaling by what the warm-up actually achieved keeps every
    // cell to about a second without making the fast ones less accurate.
    constexpr double kBudgetMicroseconds = 1.0e6;
    const Pass       warmUp              = once(std::min(nSamples, gr::Size_t{1U} << 14));

    gr::Size_t runSamples = nSamples;
    if (warmUp.elapsed.count() > 0 && warmUp.produced > 0UZ) {
        const double perMicrosecond = static_cast<double>(warmUp.consumed) / static_cast<double>(warmUp.elapsed.count());
        runSamples                  = static_cast<gr::Size_t>(std::clamp(perMicrosecond * kBudgetMicroseconds, 1.0, static_cast<double>(nSamples)));
    }

    double best = 0.0;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        const Pass pass = once(runSamples);
        if (pass.elapsed.count() > 0 && pass.produced > 0UZ) { // produced == 0 means the graph never completed a window
            best = std::max(best, static_cast<double>(pass.consumed) / static_cast<double>(pass.elapsed.count()));
        }
    }
    return best;
}

/// a single section longer than the state the block carries is refused and replaced by a pass-through, so timing
/// it would report the pass-through rather than the filter

/// a genuine second-order low-pass section: an 'a' of {1, 0, ...} is a FIR wearing an IIR's name, and comparing
/// that against a designed cascade compares two different filters
[[nodiscard]] gr::filter::FilterCoefficients<float> secondOrderSection() {
    gr::filter::FilterParameters params;
    params.order        = 2U;
    params.fLow         = 0.1;
    params.fs           = 1.0;
    const auto sections = gr::filter::iir::designFilter<float>(gr::filter::Type::LOWPASS, params, gr::filter::iir::Design::BUTTERWORTH);
    return sections.front();
}

using gr::test::kFilterLengths; // the canonical sweep every throughput table in this tree uses

/// a direct arm costs one multiply-add per tap, so a fixed sample count would spend the whole sweep in the
/// longest filter; a transform or recursion arm has no such problem and wants the full stream
struct Budget {
    gr::Size_t samples  = 0U;
    int        attempts = 1;
};

[[nodiscard]] Budget budgetFor(std::size_t nTaps, bool tapPerSample) {
    return tapPerSample ? Budget{static_cast<gr::Size_t>(gr::test::samplesForDirectFilter(nTaps)), gr::test::timingAttemptsForFilterLength(nTaps)} //
                        : Budget{static_cast<gr::Size_t>(gr::test::kStreamSamples), 3};
}

void tapSweepRow(std::string_view label, auto measureForTaps) {
    std::vector<double> rates;
    for (std::size_t nTaps : kFilterLengths) {
        rates.push_back(measureForTaps(nTaps));
    }
    std::print("  {:<38}", label);
    for (double rate : rates) {
        if (rate > 0.0) {
            std::print(" {:>8.1f}", rate);
        } else {
            std::print(" {:>8}", "--"); // the block refused this length, or no window fitted the stream
        }
    }
    std::println("");
}

} // namespace

int main() {
    std::ignore = gr::device::registerSyclRuntime();

    // what the same graph reaches with a block that only copies: a filter row approaching this one is reporting the
    // chain, not the filter, and no row can exceed it
    const double chainCeiling = megaSamplesPerSecond<gr::testing::Copy<float>>("host", [](auto&) {}, static_cast<gr::Size_t>(gr::test::kStreamSamples), 3);
    std::println("\n  chain ceiling (source -> copy -> sink, host): {:.1f} MSample/s in", chainCeiling);

    const auto header = [](std::string_view title) {
        std::println("\n  {}", title);
        std::print("  {:<38}", "block / domain");
        for (std::size_t nTaps : kFilterLengths) {
            std::print(" {:>8}", nTaps);
        }
        std::println("   (taps)");
    };

    { // where the transform overtakes the tap, on the host
        header("one low-pass, three ways -- MSample/s in, host");

        // the form a caller gets unless it names one, listed first because it is what the block's speed actually is
        tapSweepRow("fir_filter (AUTO)", [](std::size_t nTaps) {
            const Budget budget = budgetFor(nTaps, nTaps < gr::filter::fir_filter<float>::kFrequencyDomainFromTaps);
            return megaSamplesPerSecond<gr::filter::fir_filter<float>>(
                "host",
                [nTaps](auto& dut) {
                    dut.b                 = gr::Tensor<float>(lowPassTaps(nTaps));
                    dut.outputs_per_frame = frameOutputsFor(nTaps);
                },
                budget.samples, budget.attempts);
        });
        tapSweepRow("fir_filter (time)", [](std::size_t nTaps) {
            const Budget budget = budgetFor(nTaps, true);
            return megaSamplesPerSecond<gr::filter::fir_filter<float, gr::filter::ConvolutionDomain::Time>>("host", [nTaps](auto& dut) { dut.b = gr::Tensor<float>(lowPassTaps(nTaps)); }, budget.samples, budget.attempts);
        });
        tapSweepRow("fir_filter (frequency)", [](std::size_t nTaps) {
            const Budget budget = budgetFor(nTaps, false);
            return megaSamplesPerSecond<gr::filter::fir_filter<float, gr::filter::ConvolutionDomain::Frequency>>(
                "host",
                [nTaps](auto& dut) {
                    dut.b                 = gr::Tensor<float>(lowPassTaps(nTaps));
                    dut.outputs_per_frame = frameOutputsFor(nTaps);
                },
                budget.samples, budget.attempts);
        });
        tapSweepRow("BasicFilter FIR (time)", [](std::size_t nTaps) {
            const Budget budget = budgetFor(nTaps, true);
            return megaSamplesPerSecond<gr::filter::BasicFilter<float>>(
                "host",
                [nTaps](auto& dut) {
                    dut.filter_type        = gr::filter::FilterType::FIR;
                    dut.filter_domain      = gr::filter::ConvolutionDomain::Time;
                    dut.coefficient_source = gr::filter::CoefficientSource::Manual;
                    dut.a                  = gr::Tensor<float>(std::vector<float>{1.0f}); // feed-forward only
                    dut.b                  = gr::Tensor<float>(lowPassTaps(nTaps));
                },
                budget.samples, budget.attempts);
        });
        tapSweepRow("BasicFilter FIR (frequency)", [](std::size_t nTaps) {
            const Budget budget = budgetFor(nTaps, false);
            return megaSamplesPerSecond<gr::filter::BasicFilter<float>>(
                "host",
                [nTaps](auto& dut) {
                    dut.filter_type        = gr::filter::FilterType::FIR;
                    dut.filter_domain      = gr::filter::ConvolutionDomain::Frequency;
                    dut.coefficient_source = gr::filter::CoefficientSource::Manual;
                    dut.a                  = gr::Tensor<float>(std::vector<float>{1.0f}); // feed-forward only
                    dut.b                  = gr::Tensor<float>(lowPassTaps(nTaps));
                    dut.outputs_per_frame  = frameOutputsFor(nTaps);
                },
                budget.samples, budget.attempts);
        });
        tapSweepRow("iir_filter (2nd order)", [](std::size_t nTaps) {
            const Budget budget = budgetFor(nTaps, false);
            return megaSamplesPerSecond<gr::filter::iir_filter<float>>(
                "host",
                [](auto& dut) {
                    const gr::filter::FilterCoefficients<float> section = secondOrderSection();
                    dut.b                                               = gr::Tensor<float>(section.b);
                    dut.a                                               = gr::Tensor<float>(section.a);
                },
                budget.samples, budget.attempts);
        });
        std::println("  (an IIR row is flat by construction: it reaches the same corner in a handful of coefficients, whatever");
        std::println("   the FIR would have needed, which is the reason to keep it on a device rather than round-trip.");
        std::println("   The two FIR rows do not measure the same arithmetic: fir_filter declares a window and evaluates one");
        std::println("   inner product per output with no state, while BasicFilter runs the span in the transposed recursive");
        std::println("   form a cascade needs -- about twice the multiplies and a state write per tap per sample. That is the");
        std::println("   price of one block that also does IIR, and it is why its state bound bites first)");
    }

    { // where the two forms cross, finely enough to choose a default from
        constexpr std::array kCrossoverTaps{64UZ, 96UZ, 128UZ, 192UZ, 256UZ, 384UZ};

        std::println("\n  where a tap per sample gives way to a transform per frame -- MSample/s in");
        std::print("  {:<38}", "block / domain");
        for (std::size_t nTaps : kCrossoverTaps) {
            std::print(" {:>8}", nTaps);
        }
        std::println("   (taps)");

        const auto crossRow = [&]<typename TBlock>(std::string_view label, std::string_view domain, bool tapPerSample, auto prepare) {
            std::print("  {:<38}", std::format("{}, {}", label, domain));
            for (std::size_t nTaps : kCrossoverTaps) {
                const Budget budget = budgetFor(nTaps, tapPerSample);
                const double rate   = megaSamplesPerSecond<TBlock>(domain, [&](auto& dut) { prepare(dut, nTaps); }, budget.samples, budget.attempts);
                if (rate > 0.0) {
                    std::print(" {:>8.1f}", rate);
                } else {
                    std::print(" {:>8}", "--");
                }
            }
            std::println("");
        };

        for (std::string_view domain : servedDomains()) {
            crossRow.template operator()<gr::filter::fir_filter<float, gr::filter::ConvolutionDomain::Time>>("tap per sample", domain, true, //
                [](auto& dut, std::size_t nTaps) { dut.b = gr::Tensor<float>(lowPassTaps(nTaps)); });
            crossRow.template operator()<gr::filter::fir_filter<float, gr::filter::ConvolutionDomain::Frequency>>("transform per frame", domain, false, [](auto& dut, std::size_t nTaps) {
                dut.b                 = gr::Tensor<float>(lowPassTaps(nTaps));
                dut.outputs_per_frame = frameOutputsFor(nTaps);
            });
            crossRow.template operator()<gr::filter::fir_filter<float>>("AUTO (chooses)", domain, false, [](auto& dut, std::size_t nTaps) {
                dut.b                 = gr::Tensor<float>(lowPassTaps(nTaps));
                dut.outputs_per_frame = frameOutputsFor(nTaps);
            });
        }
        std::println("  (the crossing is what the AUTO form's default threshold is chosen from; it moves with the machine");
        std::println("   and with the backend, which is why naming the form explicitly overrides it)\n");
    }

    { // which of them can leave the host, and what it buys
        header("the same filters on every served domain -- MSample/s in");

        const auto domainRow = [&]<typename TBlock>(std::string_view label, std::string_view domain, bool tapPerSample, auto canMeasure, auto prepare) {
            std::print("  {:<38}", std::format("{}, {}", label, domain));
            for (std::size_t nTaps : kFilterLengths) {
                const Budget budget = budgetFor(nTaps, tapPerSample);
                const double rate   = canMeasure(nTaps) ? megaSamplesPerSecond<TBlock>(domain, [&](auto& dut) { prepare(dut, nTaps); }, budget.samples, budget.attempts) : 0.0;
                if (rate > 0.0) {
                    std::print(" {:>8.1f}", rate);
                } else {
                    std::print(" {:>8}", "--");
                }
            }
            std::println("");
        };
        constexpr auto always = [](std::size_t) { return true; };

        for (std::string_view domain : servedDomains()) {
            domainRow.template operator()<gr::filter::fir_filter<float, gr::filter::ConvolutionDomain::Time>>("fir_filter (time)", domain, true, always, //
                [](auto& dut, std::size_t nTaps) { dut.b = gr::Tensor<float>(lowPassTaps(nTaps)); });
        }
        for (std::string_view domain : servedDomains()) {
            domainRow.template operator()<gr::filter::fir_filter<float, gr::filter::ConvolutionDomain::Frequency>>("fir_filter (frequency)", domain, false, always, [](auto& dut, std::size_t nTaps) {
                dut.b                 = gr::Tensor<float>(lowPassTaps(nTaps));
                dut.outputs_per_frame = frameOutputsFor(nTaps);
            });
        }
        for (std::string_view domain : servedDomains()) {
            domainRow.template operator()<gr::filter::iir_filter<float>>("iir_filter (2nd order)", domain, false, always, [](auto& dut, std::size_t) {
                const gr::filter::FilterCoefficients<float> section = secondOrderSection();
                dut.b                                               = gr::Tensor<float>(section.b);
                dut.a                                               = gr::Tensor<float>(section.a);
            });
        }
        for (std::string_view domain : servedDomains()) {
            domainRow.template operator()<gr::filter::BasicFilter<float>>(
                "BasicFilter FIR (time)", domain, true, [](std::size_t) { return true; },
                [](auto& dut, std::size_t nTaps) {
                    dut.filter_type        = gr::filter::FilterType::FIR;
                    dut.filter_domain      = gr::filter::ConvolutionDomain::Time;
                    dut.coefficient_source = gr::filter::CoefficientSource::Manual;
                    dut.b                  = gr::Tensor<float>(lowPassTaps(nTaps));
                    dut.a                  = gr::Tensor<float>(std::vector<float>{1.0f});
                });
        }
        for (std::string_view domain : servedDomains()) {
            domainRow.template operator()<gr::filter::BasicFilter<float>>("BasicFilter IIR (2nd order, manual)", domain, false, always, [](auto& dut, std::size_t) {
                const gr::filter::FilterCoefficients<float> section = secondOrderSection();
                dut.filter_type                                     = gr::filter::FilterType::IIR;
                dut.coefficient_source                              = gr::filter::CoefficientSource::Manual;
                dut.b                                               = gr::Tensor<float>(section.b);
                dut.a                                               = gr::Tensor<float>(section.a);
            });
        }
        for (std::string_view domain : servedDomains()) {
            domainRow.template operator()<gr::filter::BasicFilter<float>>(std::format("BasicFilter IIR (order {}, designed cascade)", kIirOrder), domain, false, always, [](auto& dut, std::size_t) {
                dut.filter_type        = gr::filter::FilterType::IIR;
                dut.filter_domain      = gr::filter::ConvolutionDomain::Time;
                dut.coefficient_source = gr::filter::CoefficientSource::Designed;
                dut.filter_response    = gr::filter::Type::LOWPASS;
                dut.filter_order       = kIirOrder;
                dut.f_low              = 0.1f;
                dut.sample_rate        = 1.0f;
            });
        }
        std::println("  (a tap-per-sample arm is measured over fewer samples at the long lengths, where one pass over the");
        std::println("   full stream already takes seconds. The designed BasicFilter row is flat across the sweep because a");
        std::println("   cascade's cost is its order, not the tap count the same response would need as a FIR)\n");
    }

    return 0;
}
