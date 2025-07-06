/*  blocks/analog/test/qa_Analog.cpp
 *  ------------------------------------------------------------------
 *  Unit-tests for the tiny “analog” block-library (Boost.UT)
 *  ------------------------------------------------------------------ */
#include <boost/ut.hpp>

#include <numeric>
#include <cmath>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/analog/Analog.hpp>

using namespace gr;
using namespace gr::testing;
using namespace gr::blocks::analog;
using namespace boost::ut;

/* helper – run a source block until N samples have been produced */
template<typename Block, typename T>
std::vector<T> run_n_samples(std::size_t N [[maybe_unused]],
                             property_map cfg = {})
{
    cfg.insert_or_assign("n_samples_max", N);              // stop source

    Graph g;
    auto& src  = g.emplaceBlock<Block>(std::move(cfg));
    auto& sink = g.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_BULK>>();

    const auto rc = g.connect<"out">(src).template to<"in">(sink);
    expect(eq(rc, ConnectionResult::SUCCESS));

    scheduler::Simple sch{std::move(g)};
    expect(bool{sch.runAndWait()});

    return sink._samples;                                  // exactly N
}

/* near-equality for floats ----------------------------------------- */
static constexpr auto fnear = [](float a, float b, float tol = 1e-6f) {
    return std::fabs(a - b) <= tol;
};

/* ------------------------------------------------------------------ */
/*  test-suite                                                        */
/* ------------------------------------------------------------------ */
suite analog_basic = [] {

    "SigSource – sine wave"_test = [] {
        constexpr std::size_t N   = 8;      // one full period
        constexpr double      fs  = 8.0;
        constexpr double      f0  = 1.0;
        constexpr double      amp = 1.0;

        auto y = run_n_samples<SigSource<float>, float>(
                     N, {{"fs", fs}, {"f0", f0}, {"amp", amp}});

        expect(y.size() == N);

        for (std::size_t k = 0; k < N; ++k) {
            const float ref = static_cast<float>(amp *
                               std::sin(2.0 * M_PI * double(k) / double(N)));
            expect(fnear(y[k], ref)) << "k=" << k << " got " << y[k]
                                     << " ref " << ref;
        }
    };

    "NoiseSource – RMS around unity"_test = [] {
        constexpr std::size_t N   = 1024;
        constexpr float       tol = 0.4f;

        auto y = run_n_samples<NoiseSource<float>, float>(
                     N, {{"amp", 1.0}, {"seed", 42u}});

        expect(y.size() == N);

        const float rms = std::sqrt(std::accumulate(y.begin(), y.end(), 0.0f,
                                   [](float a, float v) { return a + v*v; }) / N);

        expect(fnear(rms, 1.0f, tol)) << " RMS=" << rms;
    };
};

int main() { /* Boost.UT runs suites before entering main */ }
