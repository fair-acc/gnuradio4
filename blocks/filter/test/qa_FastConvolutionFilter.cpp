#include <boost/ut.hpp>

#include <chrono>
#include <cmath>
#include <format>
#include <print>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/filter/FastConvolution.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include "device_test_helpers.hpp"

namespace {
using namespace gr::testing;

/// the filter the transform has to agree with: one multiply-add per tap per sample, no transform anywhere
template<typename T>
struct DirectFir : gr::Block<DirectFir<T>, gr::Resampling<>, gr::Stride<>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    std::pmr::vector<T> taps{};

    GR_MAKE_REFLECTABLE(DirectFir, in, out, taps);

    [[nodiscard]] gr::work::Status processBulk(gr::InputViewLike auto& input, gr::OutputViewLike auto& output) const noexcept {
        const std::size_t nTaps = taps.size();
        for (std::size_t n = 0UZ; n < output.size(); ++n) {
            T accumulator{};
            for (std::size_t k = 0UZ; k < nTaps; ++k) {
                accumulator += taps[k] * input[n + nTaps - 1UZ - k];
            }
            output[n] = accumulator;
        }
        return gr::work::Status::OK;
    }
};

using gr::test::servedDomains;

[[nodiscard]] std::vector<float> runFast(std::string_view domain, std::span<const float> taps, gr::Size_t outputsPerFrame, gr::Size_t nSamples, bool logSamples = true) {
    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<gr::filter::FastConvolutionFilter<float>>({{"gr:compute_domain", std::string(domain)}, {"outputs_per_frame", outputsPerFrame}, {"taps", std::vector<float>(taps.begin(), taps.end())}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", logSamples}});

    boost::ut::expect(flow.connect<"out", "in">(source, dut).has_value());
    boost::ut::expect(flow.connect<"out", "in">(dut, sink).has_value());
    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());

    std::vector<float> samples(sink._samples.size());
    for (std::size_t i = 0UZ; i < samples.size(); ++i) {
        samples[i] = sink._samples[i];
    }
    return samples;
}

[[nodiscard]] std::size_t runDirect(std::string_view domain, std::span<const float> taps, gr::Size_t outputsPerFrame, gr::Size_t nSamples) {
    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<DirectFir<float>>({{"gr:compute_domain", std::string(domain)}, //
               {"input_chunk_size", static_cast<gr::Size_t>(outputsPerFrame + taps.size() - 1UZ)}, {"output_chunk_size", outputsPerFrame}, {"stride", outputsPerFrame}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});
    dut.taps.assign(taps.begin(), taps.end());

    boost::ut::expect(flow.connect<"out", "in">(source, dut).has_value());
    boost::ut::expect(flow.connect<"out", "in">(dut, sink).has_value());
    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());
    return sink._nSamplesProduced;
}

[[nodiscard]] std::vector<float> rampTaps(std::size_t nTaps) {
    std::vector<float> taps(nTaps);
    for (std::size_t k = 0UZ; k < nTaps; ++k) {
        taps[k] = std::sin(0.3f * static_cast<float>(k)) / static_cast<float>(nTaps);
    }
    return taps;
}

template<typename TRun>
[[nodiscard]] double bestMegaSamplesPerSecond(TRun&& run, gr::Size_t nSamples, int attempts) {
    double best = 0.0;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        const auto start = std::chrono::steady_clock::now();
        run(nSamples);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        best                 = std::max(best, seconds > 0.0 ? static_cast<double>(nSamples) / seconds * 1e-6 : 0.0);
    }
    return best;
}
} // namespace

int main() {
    using namespace boost::ut;

    std::ignore = gr::device::registerSyclRuntime();

    "the block computes the filter it was given"_test = [] {
        const std::vector<float> taps = rampTaps(33UZ);
        const std::vector<float> got  = runFast("host", taps, 256U, 4096U);
        expect(gt(got.size(), 256UZ)) << "the block has to emit whole frames";

        // the source ramps, so the direct convolution is known exactly; output n is y[n + nTaps - 1]
        bool matches = true;
        for (std::size_t n = 0UZ; n < std::min<std::size_t>(200UZ, got.size()); ++n) {
            const std::size_t centre = n + taps.size() - 1UZ;
            double            exact  = 0.0;
            for (std::size_t k = 0UZ; k < taps.size(); ++k) {
                exact += static_cast<double>(taps[k]) * static_cast<double>(centre - k);
            }
            matches = matches && std::abs(static_cast<double>(got[n]) - exact) < 1e-2;
        }
        expect(matches) << "an overlap-save filter must agree with the direct convolution it replaces";
    };

    "a longer filter costs the same frame, so the shape does not change"_test = [] {
        expect(gt(runFast("host", std::vector<float>(9UZ, 1.f / 9.f), 128U, 2048U).size(), 0UZ));
        expect(gt(runFast("host", std::vector<float>(129UZ, 1.f / 129.f), 128U, 2048U).size(), 0UZ)) << "a filter longer than the requested frame must still resolve to a valid window";
    };

    "the same source computes the same filter on a device"_test = [] {
        const std::vector<float> taps = rampTaps(65UZ);
        const std::vector<float> host = runFast("host", taps, 256U, 8192U);
        expect(gt(host.size(), 0UZ)) << "the host arm has to produce something before it can be an oracle";

        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            const std::vector<float> device = runFast(domain, taps, 256U, 8192U);
            expect(eq(device.size(), host.size())) << std::format("'{}' produced a different number of samples", domain);

            // the device transform is Van Loan Stockham where the host is a butterfly, so agreement is numerical, not bitwise
            double largestDeviation = 0.0;
            double largestSample    = 1e-30;
            for (std::size_t i = 0UZ; i < std::min(host.size(), device.size()); ++i) {
                largestDeviation = std::max(largestDeviation, std::abs(static_cast<double>(device[i]) - static_cast<double>(host[i])));
                largestSample    = std::max(largestSample, std::abs(static_cast<double>(host[i])));
            }
            expect(lt(largestDeviation / largestSample, 1e-4)) << std::format("'{}' disagrees with the host by {:.3e} of full scale", domain, largestDeviation / largestSample);
        }
    };

    "what a transform per frame buys over a multiply-add per tap"_test = [] {
        std::println("  the same filter two ways, each length given the window it deserves");
        std::print("  {:<10}", "taps");
        for (std::string_view domain : servedDomains()) {
            for (std::string_view form : {"direct", "overlap"}) {
                std::print(" {:>13}", std::format("{}/{}", form, domain));
            }
        }
        std::println("");

        for (std::size_t nTaps : gr::test::kFilterLengths) {
            const std::vector<float> taps     = rampTaps(nTaps);
            const auto               window   = static_cast<gr::Size_t>(gr::test::windowForFilterLength(nTaps));
            const auto               attempts = gr::test::timingAttemptsForFilterLength(nTaps);
            std::print("  {:<10}", nTaps);
            for (std::string_view domain : servedDomains()) {
                for (std::string_view form : {"direct", "overlap"}) {
                    const bool direct = form == "direct";
                    const auto once   = [&](gr::Size_t n) {
                        if (direct) {
                            std::ignore = runDirect(domain, taps, window, n);
                        } else {
                            std::ignore = runFast(domain, taps, window, n, false);
                        }
                    };
                    const auto   samples    = static_cast<gr::Size_t>(direct ? gr::test::samplesForDirectFilter(nTaps) : gr::test::kStreamSamples);
                    const double throughput = bestMegaSamplesPerSecond(once, samples, direct ? attempts : 3);
                    expect(gt(throughput, 0.0)) << std::format("'{}' {} at {} taps produced nothing", domain, form, nTaps);
                    std::print(" {:>13.2f}", throughput);
                }
            }
            std::println("");
        }
        std::println("  (MSample/s; 'overlap' is overlap-save. The direct arm is measured over fewer samples at the");
        std::println("   long filter lengths, where a single pass over the full stream already takes seconds)\n");
    };
}
