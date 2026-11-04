#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <format>
#include <print>
#include <string_view>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include "device_dsp_chain.hpp"
#include "device_test_helpers.hpp"

/*
 * How fast the shared DSP chain runs, on the host and on every device this build serves. What it computes is
 * qa_DeviceDspChain's question; nothing here asserts a number, because a throughput is a reading, not a contract.
 */

using namespace gr::dsp::demo;
using gr::test::servedDomains;

int main() {
    setenv("OMP_NUM_THREADS", "1", 0); // forces single-thread execution, otherwise OMP is multi-threaded by default
    std::ignore = gr::device::registerSyclRuntime();

    { // cascade throughput against the frame the chain is dispatched in
        constexpr gr::Size_t kNSamples = 1U << 20;

        const auto row = [](std::string_view label, auto runFrame) {
            std::vector<double> throughput;
            for (gr::Size_t frame : {256U, 4096U, 65536U}) {
                throughput.push_back(bestMegaSamplesPerSecond([&](gr::Size_t samples) { return runFrame(frame, samples); }, kNSamples));
            }
            std::println("  {:<24} {:>10.2f} {:>10.2f} {:>10.2f}", label, throughput[0], throughput[1], throughput[2]);
        };

        std::println("\n  cascade: source -> FIR(3 taps) -> Magnitude -> sink, {} samples", kNSamples);
        std::println("  {:<24} {:>10} {:>10} {:>10}", "domain / filter", "frame 256", "frame 4k", "frame 64k");
        for (std::string_view domain : servedDomains()) {
            row(std::format("{}, whole span", domain), [domain](gr::Size_t frame, gr::Size_t samples) { return runChainOn(domain, frame, samples); });
        }
        for (std::string_view domain : servedDomains()) {
            if (domain == "host") {
                continue;
            }
            row(std::format("{}, own kernel", domain), [domain](gr::Size_t frame, gr::Size_t samples) { return runChainOn<DirectFirSycl<float>>(domain, frame, samples); });
        }
        std::println("  (MSample/s; a declared window is run per work item, so parallelism is nOut/output_chunk_size --");
        std::println("   the larger the frame, the fewer windows a span holds and the less there is to spread)\n");
    }

    { // throughput of each spike block, on every served domain
        static constexpr gr::Size_t kNSamples = 1U << 20;
        static constexpr gr::Size_t kFrame    = 1024U;

        // the rate quoted is the INPUT rate: a reduction produces one output per frame, so counting its outputs
        // would make the fastest block look like the slowest
        const auto timeGraph = [](auto&& build) {
            const auto started  = std::chrono::steady_clock::now();
            const auto produced = build();
            const auto elapsed  = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started);
            return (elapsed.count() == 0 || produced == 0UZ) ? 0.0 : static_cast<double>(kNSamples) / static_cast<double>(elapsed.count());
        };

        const auto runOne = [](std::string_view domain, auto&& emplaceDut, gr::property_map dutSettings, auto&& configureDut) {
            using namespace gr::testing;
            gr::Graph flow({{"auto_size_edges_to_chunks", true}});
            auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kNSamples}, {"mark_tag", false}});
            dutSettings.insert_or_assign("gr:compute_domain", std::string(domain));
            auto& dut  = emplaceDut(flow, std::move(dutSettings));
            auto& sink = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});
            configureDut(dut);
            std::ignore = flow.connect<"out", "in">(source, dut);
            std::ignore = flow.connect<"out", "in">(dut, sink);
            gr::scheduler::Simple<> sched;
            std::ignore = sched.exchange(std::move(flow));
            gr::test::runAbsorbingRefusal(sched);
            return static_cast<std::size_t>(sink._nSamplesProduced);
        };

        const std::vector<float> taps      = {1.f, -2.f, 0.5f};
        const std::vector<float> reference = std::vector<float>(32UZ, 0.25f);

        std::println("\n  spike blocks, {} samples, best of three after a warm-up", kNSamples);
        std::println("  {:<26} {:>12} {:>12} {:>12}", "block", "host", "host:sycl", "gpu:sycl");

        const auto row = [&](std::string_view label, auto&& runFor) {
            std::vector<double> throughput;
            for (std::string_view domain : {"host", "host:sycl", "gpu:sycl"}) {
                if (!gr::device::DeviceContextRegistry::instance().isServedExactly(domain) && domain != "host") {
                    throughput.push_back(0.0);
                    continue;
                }
                std::ignore = runFor(domain); // warm the JIT
                double best = 0.0;
                for (int attempt = 0; attempt < 3; ++attempt) {
                    best = std::max(best, timeGraph([&] { return runFor(domain); }));
                }
                throughput.push_back(best);
            }
            std::println("  {:<26} {:>12.2f} {:>12.2f} {:>12.2f}", label, throughput[0], throughput[1], throughput[2]);
        };

        row("DirectFir (3 taps)", [&](std::string_view domain) { return runOne(domain, [](gr::Graph& f, gr::property_map m) -> auto& { return f.emplaceBlock<DirectFir<float>>(std::move(m)); }, {{"input_chunk_size", kFrame + 2U}, {"output_chunk_size", kFrame}, {"stride", kFrame}}, [&](auto& dut) { dut.taps.assign(taps.begin(), taps.end()); }); });
        row("FrameRms (frame 1024)", [&](std::string_view domain) { return runOne(domain, [](gr::Graph& f, gr::property_map m) -> auto& { return f.emplaceBlock<FrameRms<float>>(std::move(m)); }, {{"input_chunk_size", kFrame}, {"output_chunk_size", gr::Size_t(1)}, {"stride", kFrame}}, [](auto&) {}); });
        row("Correlator (32 lags)", [&](std::string_view domain) { return runOne(domain, [](gr::Graph& f, gr::property_map m) -> auto& { return f.emplaceBlock<Correlator<float>>(std::move(m)); }, {{"input_chunk_size", kFrame + 31U}, {"output_chunk_size", kFrame}, {"stride", kFrame}}, [&](auto& dut) { dut.reference.assign(reference.begin(), reference.end()); }); });
        row("Magnitude (per sample)", [&](std::string_view domain) { return runOne(domain, [](gr::Graph& f, gr::property_map m) -> auto& { return f.emplaceBlock<Magnitude<float>>(std::move(m)); }, {}, [](auto&) {}); });
        std::println("  (input MSample/s; 0.00 means the domain is not served, or the block produced nothing)\n");
    }

    { // cascade throughput against filter length, where the arithmetic starts to matter
        std::println("  same chain against filter length, each length given the window it deserves");
        std::print("  {:<10}", "taps");
        for (std::string_view domain : servedDomains()) {
            std::print(" {:>12}", domain);
        }
        std::println("");

        for (std::size_t nTaps : gr::test::kFilterLengths) {
            const std::vector<float> taps    = rampTaps(nTaps);
            const auto               window  = static_cast<gr::Size_t>(gr::test::windowForFilterLength(nTaps));
            const auto               samples = static_cast<gr::Size_t>(gr::test::samplesForDirectFilter(nTaps));
            std::print("  {:<10}", nTaps);
            for (std::string_view domain : servedDomains()) {
                const auto   runChain   = [&](gr::Size_t n) { return domain == "host" ? runChainOn(domain, window, n, taps) : runChainOn<DirectFirSycl<float>>(domain, window, n, taps); };
                const double throughput = bestMegaSamplesPerSecond(runChain, samples, gr::test::timingAttemptsForFilterLength(nTaps));
                std::print(" {:>12.2f}", throughput);
            }
            std::println("");
        }
        std::println("  (MSample/s; the host arm runs the whole span, the device arms own their kernel)\n");
    }
}
