#include <boost/ut.hpp>

#include <complex>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/SubGraph.hpp>


#include <gnuradio-4.0/fourier/fft.hpp>
#include <gnuradio-4.0/testing/DeviceExpectation.hpp>
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

    auto domain = gr::makeSubGraph(std::move(inner));
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
std::vector<C> runChainedFftsInDomain(std::string_view computeDomain, bool debugFillHostRings = false) {
    gr::Graph inner;
    auto&     head = inner.emplaceBlock<FftBlock>({{"fft_size", kFftSize}, {"compute_domain", std::string(computeDomain)}});
    auto&     tail = inner.emplaceBlock<FftBlock>({{"fft_size", kFftSize}, {"compute_domain", std::string(computeDomain)}});
    expect(inner.connect<"out", "in">(head, tail).has_value());

    auto domain = gr::makeSubGraph(std::move(inner));
    expect(domain.has_value()) << [&] { return domain ? std::string{} : domain.error().message; };

    gr::Graph flow;
    auto&     src  = flow.emplaceBlock<Source>();
    auto&     sink = flow.emplaceBlock<Sink>({{"log_samples", true}});

    const std::vector<std::string> inputs  = domain->inputs;
    const std::vector<std::string> outputs = domain->outputs;
    if (debugFillHostRings) {
        std::ignore = domain->block->settings().set({{"debug_fill_host_rings", true}});
        std::ignore = domain->block->settings().applyStagedParameters();
    }
    const auto&       domainRef = flow.addBlock(std::move(domain->block));
    const std::string domainName(domainRef->uniqueName());

    expect(flow.emplaceEdge(std::string_view(src.unique_name), "out", std::string_view(domainName), inputs.at(0), gr::undefined_size, 0, "src->domain").has_value());
    expect(flow.emplaceEdge(std::string_view(domainName), outputs.at(0), std::string_view(sink.unique_name), "in", gr::undefined_size, 0, "domain->sink").has_value());

    gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::externalStep> scheduler({{"max_work_items", static_cast<std::size_t>(kFftSize)}});
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

// a domain only counts as usable if something actually serves it; a registered mock or an absent GPU must skip,
// never silently fall back to the CPU and report a device pass
[[nodiscard]] bool domainIsServed(std::string_view computeDomain) { return gr::device::DeviceContextRegistry::instance().tryResolve(computeDomain) != nullptr; }

const boost::ut::suite<"FFTSubGraph"> _fftSubGraphTests = [] {
    "an FFT inside a domain yields the same spectrum as the same FFT in a flat graph"_test = [] {
        const std::vector<C> flat     = runFlat("");
        const std::vector<C> domain   = runInDomain("");
        const std::size_t    compared = std::min(flat.size(), domain.size());

        expect(ge(compared, static_cast<std::size_t>(kFftSize))) << "both paths must deliver at least one full transform";
        expect(gt(maxMagnitude(flat, compared), 0.0)) << "guard against a vacuous pass: two all-zero spectra also compare equal";
        expect(eq(maxAbsDifference(flat, domain, compared), 0.0)) << "the host path is the same code either side, so the spectra must be bit-identical";
    };

    "a domain terminates the graph rather than running for ever"_test = [] {
        // members carry disconnect_on_done=false (their boundary peers are attached by the parent), and the domain
        // reports DONE only when every member does — so this pins down whether a domain can ever end a run
        gr::Graph inner;
        auto&     head = inner.emplaceBlock<gr::testing::Copy<C>>();
        auto&     fft  = inner.emplaceBlock<FftBlock>({{"fft_size", kFftSize}});
        expect(inner.connect<"out", "in">(head, fft).has_value());

        auto domain = gr::makeSubGraph(std::move(inner));
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

        auto domain = gr::makeSubGraph(std::move(inner));
        expect(domain.has_value());
        expect(eq(domain->inputs.size(), 1UZ)) << "the head's input is the only unclaimed input";
        expect(eq(domain->outputs.size(), 1UZ)) << "the FFT's output is the only unclaimed output";
        // named after the member and its port, so adding a member never renames a port already in use
        expect(eq(domain->inputs.at(0), std::format("{}:in", head.name)));
        expect(eq(domain->outputs.at(0), std::format("{}:out", fft.name)));
    };
};

} // namespace gr::fft_subgraph_test

int main() { /* tests are statically executed */ }
