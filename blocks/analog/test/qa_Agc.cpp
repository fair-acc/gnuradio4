#include <boost/ut.hpp>
#include <complex>
#include <numbers>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/analog/Agc.hpp>

using namespace gr;
using namespace gr::testing;
using namespace gr::blocks::analog;
using namespace boost::ut;

template<typename SampleT, typename BlockT>
std::vector<SampleT> run(const std::vector<SampleT>& drive,
                         float rate = 2.0e-2f)
{
    Graph g;

    auto& src  = g.emplaceBlock<TagSource<SampleT>>();
    src.values = drive;

    auto& agc  = g.emplaceBlock<BlockT>();
    agc.rate   = rate;

    auto& sink =
        g.emplaceBlock<TagSink<SampleT, ProcessFunction::USE_PROCESS_BULK>>();

    [[maybe_unused]] auto c1 = g.connect<"out">(src).template to<"in">(agc);
    [[maybe_unused]] auto c2 = g.connect<"out">(agc).template to<"in">(sink);

    scheduler::Simple sch{ std::move(g) };
    expect(bool{ sch.runAndWait() });

    return sink._samples;
}

suite agc = [] {

    "AgcCC tracks magnitude"_test = [] {
        constexpr std::size_t N    = 2'048;
        constexpr std::size_t skip = 1'512;          // leave 536 samples for eval

        std::vector<std::complex<float>> drive;
        drive.reserve(N);
        for (std::size_t n = 0; n < N; ++n) {
            const float phi = 2.f * std::numbers::pi_v<float> *
                              static_cast<float>(n) / static_cast<float>(N);
            drive.emplace_back(30.f * std::cos(phi), 30.f * std::sin(phi));
        }

        using AgcCC = Agc<std::complex<float>, false>;
        auto y = run<std::complex<float>, AgcCC>(drive);

        float err = 0.f;
        for (std::size_t i = skip; i < y.size(); ++i)
            err += std::fabs(std::abs(y[i]) - 1.f);
        err /= static_cast<float>(y.size() - skip);      // mean |error|

        expect(err < 0.05f);
    };

    "AgcFF tracks magnitude"_test = [] {
        constexpr std::size_t N    = 2'048;
        constexpr std::size_t skip = 1'512;

        std::vector<float> drive;
        drive.reserve(N);
        for (std::size_t n = 0; n < N; ++n) {
            const float phi = 2.f * std::numbers::pi_v<float> *
                              static_cast<float>(n) / static_cast<float>(N);
            drive.emplace_back(50.f * std::sin(phi));
        }

        using AgcFF = Agc<float, true>;
        auto y = run<float, AgcFF>(drive);

        float err = 0.f;
        for (std::size_t i = skip; i < y.size(); ++i)
            err += std::fabs(std::fabs(y[i]) - 1.f);
        err /= static_cast<float>(y.size() - skip);

        expect(err < 0.05f);
    };
};

int main() {}   // Boost.UT auto‑runs suites
