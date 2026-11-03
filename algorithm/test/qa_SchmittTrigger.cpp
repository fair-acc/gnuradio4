#include <boost/ut.hpp>

#include <cmath>
#include <limits>
#include <vector>

#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace {
using namespace gr::trigger;

struct Edge {
    EdgeDetection type;
    double        at; /// sample index plus the interpolated position
};

template<typename TTrigger, typename T>
[[nodiscard]] std::vector<Edge> edgesOf(TTrigger& trigger, const std::vector<T>& signal) {
    std::vector<Edge> found;
    for (std::size_t i = 0UZ; i < signal.size(); ++i) {
        if (const EdgeDetection edge = trigger.processOne(signal[i]); edge != EdgeDetection::NONE) {
            found.push_back({edge, static_cast<double>(i) + static_cast<double>(gr::value(trigger.lastEdgePosition))});
        }
    }
    return found;
}

void expectEdges(const std::vector<Edge>& found, const std::vector<Edge>& expected, double tolerance = 1e-5) {
    using namespace boost::ut;
    expect(eq(found.size(), expected.size())) << "number of edges";
    for (std::size_t i = 0UZ; i < std::min(found.size(), expected.size()); ++i) {
        expect(found[i].type == expected[i].type) << "edge " << i << " polarity";
        expect(approx(found[i].at, expected[i].at, tolerance)) << "edge " << i << " at " << found[i].at << ", expected " << expected[i].at;
    }
}
} // namespace

const boost::ut::suite<"SchmittTrigger thresholds"> _thresholds = [] {
    using namespace boost::ut;
    using enum EdgeDetection;
    using enum InterpolationMethod;

    "rises at level + h/2 and falls at level - h/2, both boundaries inclusive"_test = []<typename T> {
        SchmittTrigger<T> trigger({.level = T(0.5), .hysteresis = T(0.25)});
        expectEdges(edgesOf(trigger, std::vector<T>{T(0.25), T(0.45), T(0.55), T(0.625), T(1.0), T(0.45), T(0.375)}), {{RISING, 3.}, {FALLING, 6.}});
    } | std::tuple<float, double>{};

    "the state is kept inside the band: ringing between the boundaries emits nothing"_test = []<typename T> {
        SchmittTrigger<T> trigger({.level = T(0.5), .hysteresis = T(0.25)});
        expectEdges(edgesOf(trigger, std::vector<T>{T(0.0), T(0.75), T(0.45), T(0.55), T(0.4), T(0.6), T(0.45)}), {{RISING, 1.}});
    } | std::tuple<float, double>{};

    "zero hysteresis: high is x >= level, so a plateau at the level emits one edge"_test = []<typename T> {
        SchmittTrigger<T> trigger({.level = T(1), .hysteresis = T(0)});
        expectEdges(edgesOf(trigger, std::vector<T>{T(0), T(1), T(1), T(1), T(0)}), {{RISING, 1.}, {FALLING, 4.}});
    } | std::tuple<float, double>{};

    "zero hysteresis: a dip to the level from above is no edge"_test = []<typename T> {
        SchmittTrigger<T> trigger({.level = T(1), .hysteresis = T(0)});
        expectEdges(edgesOf(trigger, std::vector<T>{T(2), T(1), T(2), T(1), T(2)}), {});
        expect(trigger.state() == LineState::high) << "1 >= 1 is high";
    } | std::tuple<float, double>{};

    "the first sample settles the state without an edge"_test = []<typename T> {
        SchmittTrigger<T> trigger({.level = T(0.5), .hysteresis = T(0.25)});
        expectEdges(edgesOf(trigger, std::vector<T>{T(1.0), T(1.0), T(0.0)}), {{FALLING, 2.}});
    } | std::tuple<float, double>{};

    "the state is unknown inside the band until a boundary is reached, and reaching it is no edge"_test = []<typename T> {
        SchmittTrigger<T> trigger({.level = T(0.5), .hysteresis = T(0.25)});
        expect(trigger.state() == LineState::unknown);
        expectEdges(edgesOf(trigger, std::vector<T>{T(0.5), T(0.55)}), {});
        expect(trigger.state() == LineState::unknown) << "inside the band";
        expectEdges(edgesOf(trigger, std::vector<T>{T(0.75), T(0.25)}), {{FALLING, 1.}});
    } | std::tuple<float, double>{};

    "a reset forgets the state, so the next boundary settles it again"_test = []<typename T> {
        SchmittTrigger<T> trigger({.level = T(0.5), .hysteresis = T(0.25)});
        expectEdges(edgesOf(trigger, std::vector<T>{T(0.0), T(1.0)}), {{RISING, 1.}});
        trigger.reset();
        expect(trigger.state() == LineState::unknown);
        expectEdges(edgesOf(trigger, std::vector<T>{T(0.0), T(1.0)}), {{RISING, 1.}});
    } | std::tuple<float, double>{};

    "a non-finite sample makes the state unknown: the next boundary is no edge"_test = []<typename T> {
        SchmittTrigger<T, BASIC_LINEAR_INTERPOLATION> trigger({.level = T(0.5), .hysteresis = T(0.25)});
        const T                                       kNaN = std::numeric_limits<T>::quiet_NaN();
        const T                                       kInf = std::numeric_limits<T>::infinity();
        expectEdges(edgesOf(trigger, std::vector<T>{T(1.0), kNaN, T(0.25), T(0.75)}), {{RISING, 2.75}});
        trigger.reset();
        expectEdges(edgesOf(trigger, std::vector<T>{T(0.0), kInf, T(1.0)}), {});
    } | std::tuple<float, double>{};

    "integer types: an odd hysteresis keeps its half, rising at 2x >= 2 level + h"_test = []<typename T> {
        SchmittTrigger<T> trigger({.level = T(5), .hysteresis = T(1)});
        expectEdges(edgesOf(trigger, std::vector<T>{T(4), T(5), T(6), T(5), T(4)}), {{RISING, 2.}, {FALLING, 4.}});
    } | std::tuple<std::uint8_t, std::int16_t, std::int32_t>{};

    "integer types: zero hysteresis at the type's limits does not overflow"_test = [] {
        SchmittTrigger<std::int32_t> trigger({.level = std::numeric_limits<std::int32_t>::max(), .hysteresis = 0});
        expectEdges(edgesOf(trigger, std::vector<std::int32_t>{0, std::numeric_limits<std::int32_t>::max(), 0}), {{RISING, 1.}, {FALLING, 2.}});
    };
};

const boost::ut::suite<"SchmittTrigger crossing time"> _crossingTime = [] {
    using namespace boost::ut;
    using enum EdgeDetection;
    using enum InterpolationMethod;

    "no interpolation dates an edge at the sample that reached the boundary"_test = [] {
        SchmittTrigger<float, NO_INTERPOLATION> trigger({.level = 0.5f, .hysteresis = 0.25f});
        expectEdges(edgesOf(trigger, std::vector<float>{0.0f, 0.55f, 0.75f}), {{RISING, 2.}});
    };

    "basic interpolation dates the crossing of the boundary between the two samples either side of it"_test = []<typename T> {
        // rising boundary 0.625: from 0.5 to 1.0 it is crossed a quarter of the way, from 0.25 to 0.75 three quarters
        SchmittTrigger<T, BASIC_LINEAR_INTERPOLATION> quarter({.level = T(0.5), .hysteresis = T(0.25)});
        expectEdges(edgesOf(quarter, std::vector<T>{T(0.0), T(0.5), T(1.0)}), {{RISING, 1.25}});
        SchmittTrigger<T, BASIC_LINEAR_INTERPOLATION> threeQuarters({.level = T(0.5), .hysteresis = T(0.25)});
        expectEdges(edgesOf(threeQuarters, std::vector<T>{T(0.0), T(0.25), T(0.75)}), {{RISING, 1.75}});
        // falling boundary 0.375: from 0.5 to 0.0 it is crossed a quarter of the way
        SchmittTrigger<T, BASIC_LINEAR_INTERPOLATION> falling({.level = T(0.5), .hysteresis = T(0.25)});
        expectEdges(edgesOf(falling, std::vector<T>{T(1.0), T(0.5), T(0.0)}), {{FALLING, 1.25}});
    } | std::tuple<float, double>{};

    "basic interpolation with zero hysteresis dates the crossing of the level"_test = [] {
        SchmittTrigger<float, BASIC_LINEAR_INTERPOLATION> trigger({.level = 1.0f, .hysteresis = 0.0f});
        expectEdges(edgesOf(trigger, std::vector<float>{0.0f, 0.0f, 4.0f, 4.0f, -4.0f}), {{RISING, 1.25}, {FALLING, 3.375}});
    };

    "a small step on a large baseline keeps its crossing's fraction"_test = [] {
        // a unit step on 1e6: float resolves 1e6 + 0.5 only to 0.0625, so the crossing would be dated a sixteenth late
        SchmittTrigger<double, BASIC_LINEAR_INTERPOLATION> trigger({.level = 1e6 + 0.75, .hysteresis = 0.});
        expectEdges(edgesOf(trigger, std::vector<double>{1e6, 1e6, 1e6 + 1.}), {{RISING, 1.75}}, 1e-6);
    };

    "the crossing time does not depend on where the samples are cut"_test = [] {
        // the trigger holds its history across processOne calls: two halves of a stream give the same time as one
        SchmittTrigger<float, BASIC_LINEAR_INTERPOLATION> whole({.level = 0.5f, .hysteresis = 0.25f});
        SchmittTrigger<float, BASIC_LINEAR_INTERPOLATION> split({.level = 0.5f, .hysteresis = 0.25f});
        const std::vector<float>                          signal{0.0f, 0.1f, 0.5f, 1.0f, 0.9f};
        const auto                                        expected = edgesOf(whole, signal);
        std::vector<Edge>                                 found    = edgesOf(split, std::vector<float>(signal.begin(), signal.begin() + 3));
        for (Edge edge : edgesOf(split, std::vector<float>(signal.begin() + 3, signal.end()))) {
            found.push_back({edge.type, edge.at + 3.});
        }
        expectEdges(found, expected);
        expectEdges(expected, {{RISING, 2.25}});
    };

    "an uncertain sample's uncertainty reaches the crossing time"_test = [] {
        using U = gr::UncertainValue<float>;
        SchmittTrigger<U, BASIC_LINEAR_INTERPOLATION> trigger({.level = 0.5f, .hysteresis = 0.25f});
        expectEdges(edgesOf(trigger, std::vector<U>{U{0.0f, 0.01f}, U{0.5f, 0.01f}, U{1.0f, 0.01f}}), {{RISING, 1.25}});
        expect(gt(gr::uncertainty(trigger.lastEdgePosition), 0.0f)) << "a 1 % sample error gives a non-zero time error";
    };

    "linear regression dates the retrospective crossing of the nominal level"_test = []<typename T> {
        // a ramp of 0.1 per sample crosses the level 0.5 at sample 5; the edge is decided at 0.7 >= 0.625, sample 7
        SchmittTrigger<T, LINEAR_INTERPOLATION, 12UZ> trigger({.level = T(0.5), .hysteresis = T(0.25)});
        expectEdges(edgesOf(trigger, std::vector<T>{T(0.0), T(0.1), T(0.2), T(0.3), T(0.4), T(0.5), T(0.6), T(0.7), T(0.8)}), {{RISING, 5.}}, 1e-4);
    } | std::tuple<float, double>{};

    "a Savitzky-Golay fit dates the nominal-level crossing of a ramp to within a twentieth of a sample"_test = [] {
        using Trigger                    = SchmittTrigger<double, POLYNOMIAL_INTERPOLATION, 8UZ>;
        const auto          coefficients = Trigger::computeCoefficients(std::pmr::get_default_resource());
        Trigger             trigger({.level = 0.5, .hysteresis = 0.25});
        std::vector<double> ramp;
        for (int i = 0; i < 16; ++i) {
            ramp.push_back(0.05 * i);
        }
        trigger.setCoefficients(coefficients);
        expectEdges(edgesOf(trigger, ramp), {{RISING, 10.}}, 0.05); // the fit reflects at the window's ends, so it is not exact for a ramp
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
