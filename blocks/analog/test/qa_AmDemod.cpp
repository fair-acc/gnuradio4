#include <boost/ut.hpp>
#include <complex>
#include <numbers>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/analog/AmDemod.hpp>

using namespace gr;
using namespace gr::testing;
using namespace gr::blocks::analog;
using namespace boost::ut;

template<typename InT, typename OutT>
std::vector<OutT> run(const std::vector<InT>& drive,
                      float fs, int decim, float f_pass)
{
    Graph g;

    auto& src  = g.emplaceBlock<TagSource<InT>>();
    src.values = drive;

    auto& dem  = g.emplaceBlock<AmDemod>();
    dem.set_chan_rate(fs);
    dem.set_audio_decim(decim);
    dem.set_audio_pass(f_pass);

    auto& sink =
        g.emplaceBlock<TagSink<OutT, ProcessFunction::USE_PROCESS_BULK>>();

    [[maybe_unused]] auto c1 =
        g.connect<"out">(src).template to<"in">(dem);
    [[maybe_unused]] auto c2 =
        g.connect<"out">(dem).template to<"in">(sink);

    scheduler::Simple sch{std::move(g)};
    expect(bool{sch.runAndWait()});

    return sink._samples;
}

suite am_demod = [] {
    "constant_envelope"_test = [] {
        constexpr float fs   = 48'000.f;
        constexpr int   dec  = 8;
        constexpr float f_lp = 4'000.f;
        constexpr std::size_t N = 4 * 48'000;   // 4 s

        std::vector<std::complex<float>> drive;
        drive.reserve(N);
        for (std::size_t n = 0; n < N; ++n) {
            const float phi = 2.f * std::numbers::pi_v<float> *
                              1'000.f * static_cast<float>(n) / fs;
            drive.emplace_back(0.7f * std::cos(phi),
                               0.7f * std::sin(phi));
        }

        auto y = run<std::complex<float>, float>(drive, fs, dec, f_lp);

        const std::size_t skip = 200;           // allow IIR to settle
        float mean = 0.f;
        for (std::size_t i = skip; i < y.size(); ++i)
            mean += y[i];
        mean /= static_cast<float>(y.size() - skip);

        expect(std::abs(mean - 0.7f) < 0.005f); // ≤ 0.5 % error
    };
};

int main() {}   // Boost.UT auto‑runs suites
