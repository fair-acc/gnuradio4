#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

#include <gnuradio-4.0/algorithm/signal/PhaseLockedLoopCore.hpp>

namespace {

using gr::signal::PhaseLockedLoopCore;

[[nodiscard]] PhaseLockedLoopCore configuredLoop(double normalisedLoopBandwidth, double minPhaseStep = -std::numbers::pi, double maxPhaseStep = std::numbers::pi) {
    PhaseLockedLoopCore loop;
    loop.configure(normalisedLoopBandwidth, minPhaseStep, maxPhaseStep);
    loop.reset();
    return loop;
}

[[nodiscard]] double phaseError(double inputPhase, const PhaseLockedLoopCore& loop) { return std::remainder(inputPhase - loop.phase, 2. * std::numbers::pi); }

} // namespace

const boost::ut::suite<"PhaseLockedLoopCore"> phaseLockedLoopCoreTests = [] {
    using namespace boost::ut;

    "noise bandwidth"_test = [] {
        for (const double normalisedLoopBandwidth : {1e-4, 1e-3, 1e-2}) {
            PhaseLockedLoopCore loop         = configuredLoop(normalisedLoopBandwidth);
            double              sumOfSquares = 0.;
            const auto          nSamples     = static_cast<std::size_t>(200. / normalisedLoopBandwidth);
            for (std::size_t n = 0UZ; n < nSamples; ++n) {
                const double impulse = n == 0UZ ? 1. : 0.;
                sumOfSquares += loop.phase * loop.phase;
                loop.advance(impulse - loop.phase);
            }
            const double measuredNoiseBandwidth = sumOfSquares / 2.;
            expect(approx(measuredNoiseBandwidth / normalisedLoopBandwidth, 1., 0.02)) << "requested " << normalisedLoopBandwidth;
        }
    };

    "phase step"_test = [] {
        constexpr double    kInputPhase = 1.;
        PhaseLockedLoopCore loop        = configuredLoop(1e-3);
        for (std::size_t n = 0UZ; n < 20'000UZ; ++n) {
            loop.advance(phaseError(kInputPhase, loop));
        }
        expect(approx(phaseError(kInputPhase, loop), 0., 1e-9)) << "no remaining phase error";
        expect(approx(loop.phaseStep, 0., 1e-12)) << "no frequency change";
    };

    "frequency step"_test = [] {
        constexpr double    kInputPhaseStep = 2. * std::numbers::pi * 5e-4;
        PhaseLockedLoopCore loop            = configuredLoop(1e-3);
        double              inputPhase      = 0.3;
        for (std::size_t n = 0UZ; n < 40'000UZ; ++n) {
            loop.advance(phaseError(inputPhase, loop));
            inputPhase += kInputPhaseStep;
        }
        expect(approx(phaseError(inputPhase, loop), 0., 1e-9)) << "no remaining phase error";
        expect(approx(loop.phaseStep, kInputPhaseStep, 1e-12)) << "loop frequency equals the input frequency";
    };

    "frequency limits"_test = [] {
        constexpr double    kMaxPhaseStep = 1e-3;
        PhaseLockedLoopCore loop          = configuredLoop(1e-3, -kMaxPhaseStep, kMaxPhaseStep);
        double              inputPhase    = 0.;
        double              highestStep   = loop.phaseStep;
        for (std::size_t n = 0UZ; n < 200'000UZ; ++n) {
            loop.advance(phaseError(inputPhase, loop));
            inputPhase += 2. * kMaxPhaseStep;
            highestStep = std::max(highestStep, loop.phaseStep);
        }
        expect(le(highestStep, kMaxPhaseStep)) << "never above the upper limit";
        expect(eq(loop.phaseStep, kMaxPhaseStep)) << "held at the upper limit by an input above it";

        loop.configure(1e-3, -kMaxPhaseStep / 2., kMaxPhaseStep / 2.);
        expect(eq(loop.phaseStep, kMaxPhaseStep / 2.)) << "configure moves the loop frequency into the new range";

        loop.configure(1e-3, 2., 4.);
        loop.reset();
        expect(approx(loop.phaseStep, (2. + std::numbers::pi) / 2., 1e-15)) << "range limited to pi";
    };

    "reset"_test = [] {
        PhaseLockedLoopCore loop = configuredLoop(1e-3, 1e-3, 3e-3);
        expect(approx(loop.phaseStep, 2e-3, 1e-15)) << "starts in the middle of the range";
        loop.advance(0.5);
        loop.reset();
        expect(eq(loop.phase, 0.)) << "zero phase";
        expect(approx(loop.phaseStep, 2e-3, 1e-15)) << "middle of the range";
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
