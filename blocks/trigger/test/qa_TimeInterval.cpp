#include <boost/ut.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/math/Histogram.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/Marble.hpp>
#include <gnuradio-4.0/trigger/TimeInterval.hpp>
#include <gnuradio-4.0/trigger/ValueTrigger.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;

namespace {
template<typename T = double>
struct Measured {
    std::vector<T> values;
    gr::Size_t     unmatched = 0U;
    gr::Size_t     undated   = 0U;
};

template<typename T = double>
[[nodiscard]] Measured<T> measure(std::vector<gr::property_map> events, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     script = fixture.emplace<EventScript>();
    script._events                   = std::move(events);
    auto& interval                   = fixture.emplace<TimeInterval<T>>(std::move(settings));
    auto& sink                       = fixture.emplace<gr::testing::TagSink<T, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, interval).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(interval, sink).has_value());
    boost::ut::expect(fixture.run().has_value());
    return Measured<T>{std::vector<T>(sink._samples.begin(), sink._samples.end()), interval.n_unmatched.value, interval.n_undated.value};
}
} // namespace

const boost::ut::suite<"TimeInterval"> _timeInterval = [] {
    using namespace boost::ut;

    "each event is measured against the most recent reference"_test = [] {
        const auto found = measure({eventNamed("clk", 1'000'000'000U), eventNamed("pulse", 1'000'000'500U), eventNamed("clk", 1'001'000'000U), eventNamed("pulse", 1'001'000'250U)}, //
            {{"mode", std::string("to_reference")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(eq(found.values.size(), 2UZ)) << "one measurement per pulse";
        if (found.values.size() == 2UZ) {
            expect(approx(found.values[0], 500e-9, 1e-15)) << "500 ns after the first clock";
            expect(approx(found.values[1], 250e-9, 1e-15)) << "250 ns after the second";
        }
    };

    "a measurement with no reference yet is counted, not invented"_test = [] {
        const auto found = measure({eventNamed("pulse", 1'000'000'000U), eventNamed("clk", 1'000'001'000U), eventNamed("pulse", 1'000'002'000U)}, //
            {{"mode", std::string("to_reference")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(eq(found.values.size(), 1UZ)) << "only the pulse that followed a clock is measurable";
        expect(eq(found.unmatched, 1U)) << "and the one before it is counted as unmatched";
    };

    "a paired measurement consumes its reference, as a time of flight does"_test = [] {
        const auto found = measure({eventNamed("emit", 0U), eventNamed("echo", 1'000U), eventNamed("echo", 2'000U), eventNamed("emit", 10'000U), eventNamed("echo", 11'500U)}, //
            {{"mode", std::string("paired")}, {"reference_filter", std::string("emit")}, {"measure_filter", std::string("echo")}});

        expect(eq(found.values.size(), 2UZ)) << "one echo per emission, the second echo having no emission of its own";
        expect(eq(found.unmatched, 1U));
        if (found.values.size() == 2UZ) {
            expect(approx(found.values[0], 1e-6, 1e-15));
            expect(approx(found.values[1], 1.5e-6, 1e-15));
        }
    };

    "to_previous measures the period between successive events"_test = [] {
        const auto found = measure({eventNamed("tick", 0U), eventNamed("tick", 1'000'000U), eventNamed("tick", 2'000'000U), eventNamed("tick", 3'000'500U)}, //
            {{"mode", std::string("to_previous")}});

        expect(eq(found.values.size(), 3UZ)) << "three gaps between four events";
        if (found.values.size() == 3UZ) {
            expect(approx(found.values[0], 1e-3, 1e-12));
            expect(approx(found.values[2], 1.0005e-3, 1e-12)) << "the period that ran 500 ns long";
        }
    };

    "tie measures the departure from an ideal grid"_test = [] {
        // a 1 ms grid, the third tick arriving 500 ns late
        const auto found = measure({eventNamed("tick", 0U), eventNamed("tick", 1'000'000U), eventNamed("tick", 2'000'500U)}, //
            {{"mode", std::string("tie")}, {"nominal_period", 1e-3}});

        expect(eq(found.values.size(), 2UZ));
        if (found.values.size() == 2UZ) {
            expect(approx(found.values[0], 0., 1e-12)) << "the first tick is on the grid";
            expect(approx(found.values[1], 500e-9, 1e-12)) << "the second is 500 ns off it";
        }
    };

    "an interval beyond the limit counts as unmatched rather than reported"_test = [] {
        const auto found = measure({eventNamed("clk", 0U), eventNamed("pulse", 1'000'000'000U)}, //
            {{"mode", std::string("to_reference")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}, {"max_interval", 1e-3}});

        expect(found.values.empty()) << "a second-long gap against a millisecond limit means the pair was wrong";
        expect(eq(found.unmatched, 1U));
    };

    "an undated event is counted, never measured"_test = [] {
        std::vector<gr::property_map> events{eventNamed("clk", 1'000'000'000U), gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("pulse")}}};
        const auto                    found = measure(std::move(events), {{"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(found.values.empty());
        expect(eq(found.undated, 1U)) << "a difference against an event with no time would be invented";
    };

    "uncertainties add in quadrature when the output carries them"_test = [] {
        const auto found = measure<gr::UncertainValue<double>>({eventNamed("clk", 0U, 30U), eventNamed("pulse", 1'000U, 40U)}, //
            {{"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(eq(found.values.size(), 1UZ));
        if (!found.values.empty()) {
            expect(approx(gr::value(found.values[0]), 1e-6, 1e-15));
            expect(approx(gr::uncertainty(found.values[0]), 50e-9, 1e-15)) << "30 ns and 40 ns add to 50 ns";
        }
    };

    "an edge detector's own events measure the period between its edges"_test = [] {
        const gr::property_map values{{"lo", 1.0f}, {"hi", 5.0f}};
        const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("sync")}, //
                                              {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}},                //
                                              {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};

        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("T:lo hi lo hi lo |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&                     trigger = fixture.emplace<ValueTrigger<float, ValueCondition::level>>({{"threshold", 3.f}, {"hysteresis", 0.5f}, {"sample_rate", 1000.f}});
        trigger.in.max_samples            = 2UZ; // one edge per work call, since an event output grants a single slot per call
        auto& spent                       = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        auto& interval                    = fixture.emplace<TimeInterval<double>>({{"mode", std::string("to_previous")}});
        auto& measured                    = fixture.emplace<gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, trigger).has_value());
        expect(fixture.connect<"out", "in">(trigger, spent).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(trigger, interval).has_value());
        expect(fixture.connect<"out", "in">(interval, measured).has_value());
        expect(fixture.run().has_value());

        expect(eq(trigger.n_triggers.value, 2U)) << "two rising crossings";
        expect(eq(interval.n_undated.value, 0U)) << "both arrived dated";
        expect(eq(interval.n_intervals.value, 1U)) << "two edges give one period between them";
    };

    "what an interval measurement sees, drawn"_test = [] {
        // a 1 ms reference clock and a pulse that drifts later against it, which is how a phase is measured
        std::vector<gr::property_map> script;
        std::vector<std::uint64_t>    clocks;
        std::vector<std::uint64_t>    pulses;
        for (std::size_t tick = 0UZ; tick < 6UZ; ++tick) {
            const std::uint64_t clock = 1'000'000'000U + static_cast<std::uint64_t>(tick) * 1'000'000U;
            const std::uint64_t pulse = clock + 100'000U + static_cast<std::uint64_t>(tick) * 40'000U; // 100 us, drifting 40 us per tick
            script.push_back(eventNamed("clk", clock));
            script.push_back(eventNamed("pulse", pulse));
            clocks.push_back(clock);
            pulses.push_back(pulse);
        }
        const auto found = measure(script, {{"mode", std::string("to_reference")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(eq(found.values.size(), 6UZ)) << "one measurement per pulse";

        gr::testing::MarbleDiagram diagram{"TimeInterval: a pulse drifting against its reference"};
        auto&                      clockRow = diagram.row("clk");
        for (const std::uint64_t at : clocks) {
            clockRow.at(at, "clk");
        }
        auto& pulseRow = diagram.row("pulse");
        for (const std::uint64_t at : pulses) {
            pulseRow.at(at, "pulse");
        }
        diagram.condition("TimeInterval(to_reference)");
        diagram.print();

        std::vector<double> x(found.values.size());
        std::vector<double> y(found.values.size());
        for (std::size_t i = 0UZ; i < found.values.size(); ++i) {
            x[i] = static_cast<double>(i);
            y[i] = found.values[i] * 1e6; // microseconds, which is what an operator would read
        }
        auto chart = gr::graphs::ImChart<90, 14>({{x.front(), x.back()}, {0., 400.}});
        chart.draw(x, y, "delay [us]");
        std::println("\nthe measured delay, one point per pulse -- timing turned into a signal:");
        chart.draw();
    };

    "nearest measures against whichever reference is closer, before or after"_test = [] {
        // clocks at 0 and 1 ms; the pulse at 0.2 ms is nearest the first, the one at 0.9 ms nearest the second
        const auto found = measure({eventNamed("clk", 1'000'000'000U), eventNamed("pulse", 1'000'200'000U), eventNamed("pulse", 1'000'900'000U), eventNamed("clk", 1'001'000'000U)}, //
            {{"mode", std::string("nearest")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(eq(found.values.size(), 2UZ));
        if (found.values.size() == 2UZ) {
            expect(approx(found.values[0], 200e-6, 1e-12)) << "after its reference, so positive";
            expect(approx(found.values[1], -100e-6, 1e-12)) << "before the next one, so negative: the sign says which side";
        }
    };

    "nearest waits for the later reference rather than answering early"_test = [] {
        const auto found = measure({eventNamed("clk", 1'000'000'000U), eventNamed("pulse", 1'000'900'000U)}, //
            {{"mode", std::string("nearest")}, {"reference_filter", std::string("clk")}, {"measure_filter", std::string("pulse")}});

        expect(eq(found.values.size(), 0UZ)) << "no reference followed it, so the nearest one is not yet known";
    };

    "the distribution of a period, which is what a jitter figure is read from"_test = [] {
        constexpr std::uint64_t               kPeriod = 1'000'000U;                       // 1 ms between ticks
        constexpr std::array<std::int64_t, 5> kJitter{0, 12'000, -8'000, 4'000, -16'000}; // ns, a fixed pattern so the figures are exact
        std::vector<gr::property_map>         ticks;
        std::vector<std::uint64_t>            when;
        for (std::size_t k = 0UZ; k < 40UZ; ++k) {
            when.push_back(1'000'000'000U + static_cast<std::uint64_t>(k) * kPeriod + static_cast<std::uint64_t>(kJitter[k % kJitter.size()] + 20'000));
            ticks.push_back(eventNamed("tick", when.back()));
        }

        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = ticks;
        auto& interval                   = fixture.emplace<TimeInterval<double>>({{"mode", std::string("to_previous")}});
        auto& histogram                  = fixture.emplace<gr::blocks::math::Histogram<double>>({{"bin_min", 0.00095}, {"bin_max", 0.00105}, {"n_bins", 20U}, //
                             {"reset_on_publish", false}, {"disconnect_on_done", false}, {"axis_name", std::string("period")}, {"axis_unit", std::string("s")}});

        expect(fixture.connect<"evtOut", "evtIn">(script, interval).has_value());
        expect(fixture.connect<"out", "in">(interval, histogram).has_value());
        expect(fixture.run().has_value());

        expect(eq(histogram.n_entries.value, 39U)) << "39 gaps between 40 ticks, every one inside the range";
        expect(eq(histogram.n_underflow.value, 0U));
        expect(eq(histogram.n_overflow.value, 0U));
        expect(approx(histogram.mean.value, 1e-3, 2e-6)) << "the jitter averages out, leaving the nominal period";
        expect(histogram.stddev.value > 1e-6) << "the spread is the jitter figure itself";

        {
            gr::testing::MarbleDiagram diagram{"TimeInterval into Histogram: a period measured, then distributed"};
            auto&                      tickRow = diagram.row("tick");
            for (const std::uint64_t at : when) {
                tickRow.at(at, "tick");
            }
            diagram.condition(std::format("TimeInterval(to_previous) -> Histogram(20 bins over [0.95, 1.05] ms), mean {:.4f} ms, stddev {:.4f} ms", //
                histogram.mean.value * 1e3, histogram.stddev.value * 1e3));
            diagram.print();
        }

        std::vector<double> centres(histogram._counts.size());
        std::vector<double> counts(histogram._counts.size());
        const double        width = (0.00105 - 0.00095) / static_cast<double>(histogram._counts.size());
        for (std::size_t bin = 0UZ; bin < histogram._counts.size(); ++bin) {
            centres[bin] = (0.00095 + (static_cast<double>(bin) + 0.5) * width) * 1e3; // ms, which is what an operator would read
            counts[bin]  = static_cast<double>(histogram._counts[bin]);
        }
        auto chart = gr::graphs::ImChart<90, 14>({{centres.front(), centres.back()}, {0., *std::ranges::max_element(counts)}});
        chart.draw<gr::graphs::Style::Bars>(centres, counts, "counts");
        std::println("\nthe distribution of the measured period -- the jitter histogram:");
        chart.draw();
    };
};

int main() { /* tests are statically executed */ }
