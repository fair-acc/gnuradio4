#include <boost/ut.hpp>
#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/analog/FmDet.hpp>        
#include <gnuradio-4.0/analog/FrequencyMod.hpp> // to build an end‑to‑end loop

using namespace gr;
using namespace gr::testing;
using namespace gr::blocks::analog;
using namespace boost::ut;

template<typename InT , typename OutT>
std::vector<OutT> run_pipeline(const std::vector<InT>& drive,
                               property_map det_cfg = {})
{
    Graph g;
    auto& src  = g.emplaceBlock<TagSource<InT>>(
        {{"values", drive}, {"n_samples_max", drive.size()}});
    auto& det  = g.emplaceBlock<FmDet<std::complex<float>>>(std::move(det_cfg));
    auto& sink = g.emplaceBlock<TagSink<OutT, ProcessFunction::USE_PROCESS_BULK>>();

    expect(eq(g.connect<"out">(src ).template to<"in">(det ), ConnectionResult::SUCCESS));
    expect(eq(g.connect<"out">(det ).template to<"in">(sink), ConnectionResult::SUCCESS));

    scheduler::Simple sch{ std::move(g) };
    expect(bool{ sch.runAndWait() });
    return sink._samples;
}

static constexpr auto feq = [](float a, float b, float tol = 1e-6f) {
    return std::fabs(a - b) <= tol;
};

suite fm_det = [] {

    "scale / freq‑range setters"_test = [] {
        FmDet<std::complex<float>> det;

        det.set_freq_range(1.0f, 2.0f);
        expect(feq(det.freq_low (), 1.0f));
        expect(feq(det.freq_high(), 2.0f));

        det.set_scale(4.0f);
        expect(feq(det.scale(), 4.0f));

        /* bias = ½·scl·(hi+lo)/(hi‑lo) = ½·4·3 / 1 = 6 */
        expect(feq(det.bias(), 6.0f));
    };

    "end‑to‑end FM  → slope‑det"_test = [] {
        constexpr float  f0   = 0.125f;                   // Hz (matches sens so y≈1)
        constexpr float  fs   [[maybe_unused]] = 1.0f;    // kept for clarity
        constexpr float  sens = std::numbers::pi_v<float> / 4.0f;
        constexpr float  gain = 1.0f / sens;
        constexpr size_t N    = 100;

        std::vector<std::complex<float>> drive;
        drive.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            float phase = 2.0f * std::numbers::pi_v<float> * f0 *
                          static_cast<float>(i);
            drive.emplace_back(std::cos(phase), std::sin(phase));
        }

        auto y = run_pipeline<std::complex<float>, float>(
                     drive, {{"scl", gain}});   // detector scale

        /* first sample undefined (prev‑sample = 0) – ignore */
        expect(y.size() == N);
        for (size_t i = 1; i < N; ++i)
            expect(feq(y[i], 1.0f, 5e-2f)) << "k=" << i << " got " << y[i];
    };
};

int main() { }   // Boost.UT launches suites before entering main
