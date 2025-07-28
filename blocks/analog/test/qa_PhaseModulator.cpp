#include <boost/ut.hpp>
#include <complex>
#include <numbers>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/analog/PhaseModulator.hpp>

using namespace gr;
using namespace gr::testing;
using namespace gr::blocks::analog;
using namespace boost::ut;

template<typename OutT = std::complex<float>>
std::vector<OutT> run(const std::vector<float>& drive, float sens)
{
    Graph g;

    auto& src  = g.emplaceBlock<TagSource<float>>();
    src.values = drive;                        // emit exactly once

    auto& mod  = g.emplaceBlock<PhaseModulator>();
    mod.set_sensitivity(sens);

    auto& sink =
        g.emplaceBlock<TagSink<OutT, ProcessFunction::USE_PROCESS_BULK>>();

    [[maybe_unused]] auto _c1 =
        g.connect<"out">(src).template to<"in">(mod);
    [[maybe_unused]] auto _c2 =
        g.connect<"out">(mod).template to<"in">(sink);

    scheduler::Simple sch{ std::move(g) };
    expect(bool{ sch.runAndWait() });

    return sink._samples;
}

suite phase_mod = [] {
    "basic"_test = [] {
        constexpr float sens = std::numbers::pi_v<float> / 4.f;

        const std::vector<float> drive = { 0.25f, 0.5f, 0.25f,
                                           -0.25f, -0.5f, -0.25f };

        std::vector<std::complex<float>> ref;
        ref.reserve(drive.size());
        for (auto v : drive) {
            const float phi = sens * v;
            ref.emplace_back(std::cos(phi), std::sin(phi));
        }

        auto y = run(drive, sens);

        expect(y.size() >= ref.size());

        constexpr float tol = 1e-5f;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            expect(std::abs(y[i].real() - ref[i].real()) < tol);
            expect(std::abs(y[i].imag() - ref[i].imag()) < tol);
        }
    };
};

int main() {}   // Boost.UT auto‑runs suites
