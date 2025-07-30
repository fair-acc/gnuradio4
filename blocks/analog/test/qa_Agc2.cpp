#include <boost/ut.hpp>
#include <complex>
#include <numbers>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/analog/Agc2.hpp>

using namespace gr;
using namespace gr::testing;
using namespace gr::blocks::analog;
using namespace boost::ut;

template<typename SampleT, typename BlockT>
std::vector<SampleT> run(const std::vector<SampleT>& drive,
                         float a_rate = 1.0e-1f,
                         float d_rate = 1.0e-2f)
{
    Graph g;

    auto& src  = g.emplaceBlock<TagSource<SampleT>>();
    src.values = drive;

    auto& agc2 = g.emplaceBlock<BlockT>();
    agc2.attack_rate = a_rate;
    agc2.decay_rate  = d_rate;

    auto& sink =
        g.emplaceBlock<TagSink<SampleT, ProcessFunction::USE_PROCESS_BULK>>();

    (void)g.connect<"out">(src).template to<"in">(agc2);
    (void)g.connect<"out">(agc2).template to<"in">(sink);

    scheduler::Simple sch{ std::move(g) };
    expect(bool{ sch.runAndWait() });

    return sink._samples;
}

suite agc2 = [] {

    "Agc2CC converges"_test = [] {
        constexpr std::size_t N    = 3072;
        constexpr std::size_t skip = 2048;

        std::vector<std::complex<float>> drive;
        drive.reserve(N);
        for (std::size_t n = 0; n < N; ++n) {
            const float phi = 2.f * std::numbers::pi_v<float> *
                              static_cast<float>(n) / static_cast<float>(N);
            drive.emplace_back(40.f * std::sin(phi),
                               40.f * std::cos(phi));
        }

        auto y = run<std::complex<float>, Agc2CC>(drive);

        float err = 0.f;
        for (std::size_t i = skip; i < y.size(); ++i)
            err += std::fabs(std::abs(y[i]) - 1.f);
        err /= static_cast<float>(y.size() - skip);

        expect(err < 0.05f);
    };

    "Agc2FF converges"_test = [] {
        constexpr std::size_t N    = 3072;
        constexpr std::size_t skip = 2048;

        std::vector<float> drive;
        drive.reserve(N);
        for (std::size_t n = 0; n < N; ++n) {
            const float phi = 2.f * std::numbers::pi_v<float> *
                              static_cast<float>(n) / static_cast<float>(N);
            drive.emplace_back(60.f * std::cos(phi));
        }

        auto y = run<float, Agc2FF>(drive);

        float err = 0.f;
        for (std::size_t i = skip; i < y.size(); ++i)
            err += std::fabs(std::fabs(y[i]) - 1.f);
        err /= static_cast<float>(y.size() - skip);

        expect(err < 0.05f);
    };
};

int main() {}   // Boost.UT auto‑runs suites
