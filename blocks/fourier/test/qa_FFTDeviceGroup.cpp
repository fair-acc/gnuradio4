#include <boost/ut.hpp>

#include <bit>
#include <complex>
#include <vector>

#include <gnuradio-4.0/DeviceGroup.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>

#include <gnuradio-4.0/fourier/fft.hpp>
#include <gnuradio-4.0/test/DeviceExpectation.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

// The vertical stack: a real DSP block, not a Copy, running inside a device domain in a scheduled graph. The claim
// under test is that routing an FFT through a domain changes nothing an observer can see — the same input yields
// the same spectrum whether the block sits in the flat graph or inside a domain that drives it from its own work().

namespace gr::fft_subgraph_test {

using namespace boost::ut;
using namespace gr;

using C        = std::complex<float>;
using FftBlock = gr::blocks::fft::FFT<float>; // stream mode: complex in, complex out
using Source   = gr::testing::CountingSource<C>;
using Sink     = gr::testing::TagSink<C, gr::testing::ProcessFunction::USE_PROCESS_BULK>;

constexpr gr::Size_t  kFftSize    = 1024U;
constexpr std::size_t kSteps      = 64UZ;
constexpr std::size_t kChainSteps = 16UZ; // shorter than kSteps: a busy GPU only needs a handful of dispatches to prove the count

std::vector<C> runFlat(std::string_view computeDomain) {
    gr::Graph flow;
    auto&     src  = flow.emplaceBlock<Source>();
    auto&     fft  = flow.emplaceBlock<FftBlock>({{"fft_size", kFftSize}, {"compute_domain", std::string(computeDomain)}});
    auto&     sink = flow.emplaceBlock<Sink>({{"log_samples", true}});
    expect(flow.connect<"out", "in">(src, fft).has_value());
    expect(flow.connect<"out", "in">(fft, sink).has_value());

    gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::externalStep> scheduler;
    expect(scheduler.exchange(std::move(flow)).has_value());
    expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value());
    expect(scheduler.changeStateTo(lifecycle::State::RUNNING).has_value());
    for (std::size_t i = 0UZ; i < kSteps; ++i) {
        std::ignore = scheduler.step();
    }
    return std::vector<C>(sink._samples.begin(), sink._samples.end());
}

std::vector<C> runInDomain(std::string_view computeDomain) {
    gr::Graph inner;
    auto&     head = inner.emplaceBlock<gr::testing::Copy<C>>();
    auto&     fft  = inner.emplaceBlock<FftBlock>({{"fft_size", kFftSize}, {"compute_domain", std::string(computeDomain)}});
    expect(inner.connect<"out", "in">(head, fft).has_value());

    auto domain = gr::makeDeviceGroup(std::move(inner));
    expect(domain.has_value()) << [&] { return domain ? std::string{} : domain.error().message; };

    gr::Graph flow;
    auto&     src  = flow.emplaceBlock<Source>();
    auto&     sink = flow.emplaceBlock<Sink>({{"log_samples", true}});

    const std::vector<std::string> inputs    = domain->inputs;
    const std::vector<std::string> outputs   = domain->outputs;
    const auto&                    domainRef = flow.addBlock(std::move(domain->block));
    const std::string              domainName(domainRef->uniqueName());

    expect(flow.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(domainName), inputs.at(0), gr::undefined_size, 0, "src->domain").has_value());
    expect(flow.emplaceEdge(std::string_view(domainName), outputs.at(0), std::string_view(sink.unique_name), "in", gr::undefined_size, 0, "domain->sink").has_value());

    gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::externalStep> scheduler;
    expect(scheduler.exchange(std::move(flow)).has_value());
    expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value());
    expect(scheduler.changeStateTo(lifecycle::State::RUNNING).has_value());
    for (std::size_t i = 0UZ; i < kSteps; ++i) {
        std::ignore = scheduler.step();
    }
    return std::vector<C>(sink._samples.begin(), sink._samples.end());
}

// both members share one compute_domain, so the head->tail hop stays inside the group. max_work_items caps every
// block's work() to exactly one fft_size chunk per call, which is what makes the dispatch count below knowable.
std::vector<C> runChainedFftsInDomain(std::string_view computeDomain) {
    gr::Graph inner;
    auto&     head = inner.emplaceBlock<FftBlock>({{"fft_size", kFftSize}, {"compute_domain", std::string(computeDomain)}});
    auto&     tail = inner.emplaceBlock<FftBlock>({{"fft_size", kFftSize}, {"compute_domain", std::string(computeDomain)}});
    expect(inner.connect<"out", "in">(head, tail).has_value());

    auto domain = gr::makeDeviceGroup(std::move(inner));
    expect(domain.has_value()) << [&] { return domain ? std::string{} : domain.error().message; };

    gr::Graph flow;
    auto&     src  = flow.emplaceBlock<Source>();
    auto&     sink = flow.emplaceBlock<Sink>({{"log_samples", true}});

    const std::vector<std::string> inputs    = domain->inputs;
    const std::vector<std::string> outputs   = domain->outputs;
    const auto&                    domainRef = flow.addBlock(std::move(domain->block));
    const std::string              domainName(domainRef->uniqueName());

    expect(flow.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(domainName), inputs.at(0), gr::undefined_size, 0, "src->domain").has_value());
    expect(flow.emplaceEdge(std::string_view(domainName), outputs.at(0), std::string_view(sink.unique_name), "in", gr::undefined_size, 0, "domain->sink").has_value());

    gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::externalStep> scheduler({{"max_work_items", gr::Size_t(kFftSize)}});
    expect(scheduler.exchange(std::move(flow)).has_value());
    expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value());
    expect(scheduler.changeStateTo(lifecycle::State::RUNNING).has_value());
    for (std::size_t i = 0UZ; i < kChainSteps; ++i) {
        std::ignore = scheduler.step();
    }
    return std::vector<C>(sink._samples.begin(), sink._samples.end());
}

[[nodiscard]] double maxMagnitude(const std::vector<C>& values, std::size_t n) {
    double peak = 0.0;
    for (std::size_t i = 0UZ; i < n; ++i) {
        peak = std::max(peak, static_cast<double>(std::abs(values[i])));
    }
    return peak;
}

[[nodiscard]] double maxAbsDifference(const std::vector<C>& lhs, const std::vector<C>& rhs, std::size_t n) {
    double worst = 0.0;
    for (std::size_t i = 0UZ; i < n; ++i) {
        worst = std::max(worst, static_cast<double>(std::abs(lhs[i] - rhs[i])));
    }
    return worst;
}

const auto fftDeviceGroupTests = [] {
    using gr::testing::operator""_domain_test;

    "an FFT inside a domain yields the same spectrum as the same FFT in a flat graph"_test = [] {
        const std::vector<C> flat     = runFlat("");
        const std::vector<C> domain   = runInDomain("");
        const std::size_t    compared = std::min(flat.size(), domain.size());

        expect(ge(compared, static_cast<std::size_t>(kFftSize))) << "both paths must deliver at least one full transform";
        expect(gt(maxMagnitude(flat, compared), 0.0)) << "guard against a vacuous pass: two all-zero spectra also compare equal";
        expect(eq(maxAbsDifference(flat, domain, compared), 0.0)) << "the host path is the same code either side, so the spectra must be bit-identical";
    };

    // a float radix-2 FFT rounds by about log2(N) ulps of its largest bin, about 6e-7 of it at N = 1024: 1e-5 of the peak bounds a device's spectrum
    constexpr double kDeviceTolerance = 1e-5;

    "an FFT on a device yields the host's spectrum, in a flat graph and inside a domain"_domain_test = [](auto&, std::string_view domain) {
        const std::string    placed           = std::string(domain) + "!";
        const std::vector<C> host             = runFlat("");
        const std::uint64_t  dispatchedBefore = gr::testing::gpuDispatchesOn(domain);
        const std::vector<C> flat             = runFlat(placed);
        const std::vector<C> grouped          = runInDomain(placed);
        const std::size_t    compared         = std::min({host.size(), flat.size(), grouped.size()});
        const double         peak             = maxMagnitude(host, compared);

        expect(ge(compared, static_cast<std::size_t>(kFftSize))) << domain << ": every path must deliver at least one full transform";
        expect(le(maxAbsDifference(host, flat, compared), kDeviceTolerance * peak)) << domain << ": in a flat graph";
        expect(le(maxAbsDifference(host, grouped, compared), kDeviceTolerance * peak)) << domain << ": inside a domain";
        expect(eq(gr::testing::gpuDispatchesOn(domain), dispatchedBefore)) << domain << ": spectra a host sink reads stay with SimdFFT, so no kernel runs";
    } | gr::testing::overGraphs(gr::testing::kOffloadDomains);

    "two FFTs chained inside a domain on a device yield the host chain's spectrum"_domain_test = [](auto&, std::string_view domain) {
        const std::vector<C> host             = runChainedFftsInDomain("");
        const std::uint64_t  dispatchedBefore = gr::testing::gpuDispatchesOn(domain);
        const std::vector<C> device           = runChainedFftsInDomain(std::string(domain) + "!");
        const std::uint64_t  dispatched       = gr::testing::gpuDispatchesOn(domain) - dispatchedBefore;
        const std::size_t    compared         = std::min(host.size(), device.size());
        const double         peak             = maxMagnitude(host, compared);

        expect(ge(compared, static_cast<std::size_t>(kFftSize))) << domain << ": the chain must deliver at least one full transform";
        expect(gt(peak, 0.0)) << "guard against a vacuous pass";
        expect(le(maxAbsDifference(host, device, compared), 2.0 * kDeviceTolerance * peak)) << domain << ": two transforms, twice the rounding";
        const bool webGpu = domain == "gpu:webgpu" || domain == "gpu:swift";
        expect(!webGpu || dispatched >= static_cast<std::uint64_t>((std::countr_zero(kFftSize) + 2) / 3)) << domain << ": " << dispatched << " kernels ran; the head's spectra stay in the domain, so its Stockham passes, up to three stages each, run on the GPU";
    } | gr::testing::overGraphs(gr::testing::kOffloadDomains);

    "a domain terminates the graph rather than running for ever"_test = [] {
        // members carry disconnect_on_done=false (their boundary peers are attached by the parent), and the domain
        // reports DONE only when every member does — so this pins down whether a domain can ever end a run
        gr::Graph inner;
        auto&     head = inner.emplaceBlock<gr::testing::Copy<C>>();
        auto&     fft  = inner.emplaceBlock<FftBlock>({{"fft_size", kFftSize}});
        expect(inner.connect<"out", "in">(head, fft).has_value());

        auto domain = gr::makeDeviceGroup(std::move(inner));
        expect(domain.has_value());

        gr::Graph flow;
        auto&     src  = flow.emplaceBlock<Source>({{"n_samples_max", 4U * kFftSize}});
        auto&     sink = flow.emplaceBlock<Sink>({{"log_samples", false}});

        const std::vector<std::string> inputs    = domain->inputs;
        const std::vector<std::string> outputs   = domain->outputs;
        const auto&                    domainRef = flow.addBlock(std::move(domain->block));
        const std::string              domainName(domainRef->uniqueName());
        expect(flow.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(domainName), inputs.at(0), gr::undefined_size, 0, "src->domain").has_value());
        expect(flow.emplaceEdge(std::string_view(domainName), outputs.at(0), std::string_view(sink.unique_name), "in", gr::undefined_size, 0, "domain->sink").has_value());

        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::externalStep> scheduler;
        expect(scheduler.exchange(std::move(flow)).has_value());
        expect(scheduler.changeStateTo(lifecycle::State::INITIALISED).has_value());
        expect(scheduler.changeStateTo(lifecycle::State::RUNNING).has_value());

        constexpr std::size_t kCap  = 512UZ;
        bool                  ended = false;
        for (std::size_t i = 0UZ; i < kCap && !ended; ++i) {
            ended = scheduler.step().status == work::Status::DONE;
        }
        expect(ended) << "a bounded source must be able to end a run that goes through a domain; without this a "
                         "domain can only ever be driven by a step budget, never by runAndWait()";
    };

    "a domain carrying an FFT keeps the block's own port and settings contract"_test = [] {
        gr::Graph inner;
        auto&     head = inner.emplaceBlock<gr::testing::Copy<C>>();
        auto&     fft  = inner.emplaceBlock<FftBlock>({{"fft_size", kFftSize}});
        expect(inner.connect<"out", "in">(head, fft).has_value());

        auto domain = gr::makeDeviceGroup(std::move(inner));
        expect(domain.has_value());
        expect(eq(domain->inputs.size(), 1UZ)) << "the head's input is the only unclaimed input";
        expect(eq(domain->outputs.size(), 1UZ)) << "the FFT's output is the only unclaimed output";
        // named after the member and its port, so adding a member never renames a port already in use
        expect(eq(domain->inputs.at(0), std::format("{}:in", head.name)));
        expect(eq(domain->outputs.at(0), std::format("{}:out", fft.name)));
    };
};

} // namespace gr::fft_subgraph_test

int main() {
    using boost::ut::operator""_test;
    "FFTDeviceGroup"_test = gr::fft_subgraph_test::fftDeviceGroupTests;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
