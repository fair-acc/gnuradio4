#include <boost/ut.hpp>

#include <numeric>
#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/math/Histogram.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

using namespace gr::blocks::math;
using gr::testing::ProcessFunction;
using gr::testing::TagSource;

namespace {
/// publishes a fixed list of events, so which of them the histogram acts on does not depend on work-call granularity
struct EventScript : gr::Block<EventScript> {
    /// an event output claims one slot per work call unless told otherwise, and a script must place all of its events
    /// before the stream that drives the graph ends
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    GR_MAKE_REFLECTABLE(EventScript, evtOut);

    std::vector<gr::property_map> _events;
    std::size_t                   _published = 0UZ;

    gr::work::Status processBulk(gr::OutputSpanLike auto& evtSpan) {
        std::size_t emitted = 0UZ;
        while (_published < _events.size() && emitted < evtSpan.size()) {
            if (!gr::emitEvent(evtSpan, emitted, gr::property_map_view{_events[_published]})) {
                break;
            }
            ++_published;
            ++emitted;
        }
        evtSpan.publish(emitted);
        return gr::work::Status::OK; // the stream decides when the graph ends, not the event source
    }
};

struct SetSink : gr::Block<SetSink> {
    gr::PortIn<gr::DataSet<double>> in;

    GR_MAKE_REFLECTABLE(SetSink, in);

    std::vector<gr::DataSet<double>> _sets;

    void processOne(gr::DataSet<double> set) { _sets.push_back(std::move(set)); }
};

[[nodiscard]] gr::property_map eventNamed(std::string name) { return gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::move(name)}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t(1'000'000'000U)}}; }

[[nodiscard]] double totalOf(const gr::DataSet<double>& set) { return std::accumulate(set.signal_values.begin(), set.signal_values.end(), 0.); }

constexpr gr::Size_t kSamples = 500U;
} // namespace

const boost::ut::suite<"Histogram"> _histogram = [] {
    using namespace boost::ut;

    const std::vector<double> fiveBins{0.1, 0.3, 0.5, 0.7, 0.9}; // one value per bin of a five-bin unit range

    "each value is counted in the bin that holds it"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<TagSource<double>>({{"n_samples_max", kSamples}, {"values", fiveBins}});
        auto&                     histogram = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 5U}, {"reset_on_publish", false}, {"disconnect_on_done", false}});

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.run().has_value());

        expect(eq(histogram.n_entries.value, kSamples)) << "every sample of the run is inside the range";
        expect(eq(histogram.n_underflow.value, 0U));
        expect(eq(histogram.n_overflow.value, 0U));
        for (const std::uint64_t counted : histogram._counts) {
            expect(eq(counted, std::uint64_t{kSamples / 5U})) << "the five values are evenly spread, so the five bins are too";
        }
    };

    "a value outside the range is counted rather than clamped into an end bin"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<TagSource<double>>({{"n_samples_max", 90U}, {"values", std::vector<double>{-0.5, 0.5, 1.5}}});
        auto&                     histogram = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 4U}, {"reset_on_publish", false}, {"disconnect_on_done", false}});

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.run().has_value());

        expect(eq(histogram.n_underflow.value, 30U));
        expect(eq(histogram.n_overflow.value, 30U));
        expect(eq(histogram.n_entries.value, 30U)) << "only the third of the samples that is in range is an entry";
        expect(eq(std::accumulate(histogram._counts.begin(), histogram._counts.end(), std::uint64_t{0}), std::uint64_t{30}));
        expect(eq(histogram._counts.front(), std::uint64_t{0})) << "an underflow must not pile up in the first bin";
        expect(eq(histogram._counts.back(), std::uint64_t{0})) << "nor an overflow in the last";
    };

    "the figures come from the values, not from the bin centres"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<TagSource<double>>({{"n_samples_max", kSamples}, {"values", fiveBins}});
        auto&                     histogram = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 2U}, {"reset_on_publish", false}, {"disconnect_on_done", false}});

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.run().has_value());

        expect(approx(histogram.mean.value, 0.5, 1e-9)) << "two bins would have put the mean at 0.25 or 0.75";
        expect(approx(histogram.stddev.value, 0.2829, 1e-3)) << "the sample deviation of {0.1 .. 0.9}";
        expect(approx(histogram.min_value.value, 0.1, 1e-9));
        expect(approx(histogram.max_value.value, 0.9, 1e-9));
    };

    "the filter decides which event publishes a snapshot"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TagSource<double>>({{"n_samples_max", kSamples}, {"values", fiveBins}});
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {eventNamed("other"), eventNamed("snapshot")};
        auto& histogram                  = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 5U}, {"trigger_name", std::string("snapshot")}});
        auto& sets                       = fixture.emplace<SetSink>();

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, histogram, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.connect<"out", "in">(histogram, sets).has_value());
        expect(fixture.run().has_value());

        expect(eq(histogram.n_published.value, 1U)) << "one of the two events is named by the filter, the other is not";
        expect(eq(sets._sets.size(), 1UZ));
        if (!sets._sets.empty()) {
            expect(eq(sets._sets[0].axis_values[0].size(), 5UZ)) << "the axis carries one centre per bin";
            expect(approx(sets._sets[0].axis_values[0][0], 0.1, 1e-9)) << "the centre of the first bin of a unit range in five";
            expect(sets._sets[0].meta_information[0].contains("stddev")) << "the figures travel with the picture";
        }
    };

    "an unnamed filter lets any event publish"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TagSource<double>>({{"n_samples_max", kSamples}, {"values", fiveBins}});
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {eventNamed("whatever")};
        auto& histogram                  = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 5U}});
        auto& sets                       = fixture.emplace<SetSink>();

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, histogram, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.connect<"out", "in">(histogram, sets).has_value());
        expect(fixture.run().has_value());

        expect(eq(histogram.n_published.value, 1U));
        expect(eq(sets._sets.size(), 1UZ));
    };

    "a periodic snapshot counts each sample once"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<TagSource<double>>({{"n_samples_max", kSamples}, {"values", fiveBins}});
        auto&                     histogram = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 5U}, {"n_samples", 100U}});
        auto&                     sets      = fixture.emplace<SetSink>();

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.connect<"out", "in">(histogram, sets).has_value());
        expect(fixture.run().has_value());

        expect(ge(sets._sets.size(), 1UZ)) << "500 samples at 100 a snapshot publishes at least once";
        const double published = std::accumulate(sets._sets.begin(), sets._sets.end(), 0., [](double sum, const gr::DataSet<double>& set) { return sum + totalOf(set); });
        expect(approx(published + static_cast<double>(histogram.n_entries.value), static_cast<double>(kSamples), 1e-9)) << "a reset snapshot covers the samples since the last one: none counted twice, none lost";
    };

    "a cumulative snapshot keeps what the previous one counted"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<TagSource<double>>({{"n_samples_max", kSamples}, {"values", fiveBins}});
        auto&                     histogram = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 5U}, {"n_samples", 100U}, {"reset_on_publish", false}});
        auto&                     sets      = fixture.emplace<SetSink>();

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.connect<"out", "in">(histogram, sets).has_value());
        expect(fixture.run().has_value());

        expect(ge(sets._sets.size(), 1UZ));
        for (std::size_t i = 1UZ; i < sets._sets.size(); ++i) {
            expect(ge(totalOf(sets._sets[i]), totalOf(sets._sets[i - 1UZ]))) << "a cumulative histogram never loses an entry";
        }
        expect(eq(histogram.n_entries.value, kSamples));
    };

    "a bin count of zero and an inverted range are refused"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     histogram = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 5U}});
        histogram.settings().init();
        std::ignore = histogram.settings().applyStagedParameters(); // what the constructor asked for
        expect(eq(histogram.n_bins.value, 5U));

        expect(histogram.settings().set({{"n_bins", gr::Size_t(0)}, {"bin_max", -1.}}).empty());
        std::ignore = histogram.settings().activateContext();
        std::ignore = histogram.settings().applyStagedParameters();

        expect(eq(histogram.n_bins.value, 5U)) << "a histogram of no bins counts nothing, so the previous count stays";
        expect(approx(histogram.bin_max.value, 1., 1e-12)) << "an inverted range has no bin to fill, so the previous edge stays";
    };

    "a setting that leaves the bins alone keeps what was counted"_test = [] {
        Histogram<double> histogram({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 5U}});
        histogram.settings().init();
        std::ignore = histogram.settings().applyStagedParameters();
        histogram.start();
        expect(eq(histogram._counts.size(), 5UZ)) << "the bins the constructor asked for";
        for (std::size_t i = 0UZ; i < 10UZ; ++i) {
            histogram._histogram.add(0.5);
        }
        expect(eq(histogram._histogram.entries, std::uint64_t{10}));

        expect(histogram.settings().set({{"axis_unit", std::string("ms")}}).empty());
        std::ignore = histogram.settings().activateContext();
        std::ignore = histogram.settings().applyStagedParameters();
        expect(eq(histogram._histogram.entries, std::uint64_t{10})) << "settingsChanged fires on any setting, and must not throw an accumulation away";

        expect(histogram.settings().set({{"n_bins", gr::Size_t(10)}}).empty());
        std::ignore = histogram.settings().activateContext();
        std::ignore = histogram.settings().applyStagedParameters();
        expect(eq(histogram._histogram.entries, std::uint64_t{0})) << "a change of shape has no counts to keep";
        expect(eq(histogram._counts.size(), 10UZ));
        expect(eq(histogram._histogram.bins.size(), 10UZ)) << "and the accumulator views the new storage, not the old";
    };

    "the distribution a run produced, drawn"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TagSource<double>>({{"n_samples_max", 1000U}, {"values", std::vector<double>{0.35, 0.45, 0.45, 0.5, 0.5, 0.5, 0.55, 0.55, 0.65, 0.5}}});
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {eventNamed("snapshot")};
        auto& histogram                  = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 20U}, {"trigger_name", std::string("snapshot")}, {"reset_on_publish", false}, {"axis_name", std::string("interval")}, {"axis_unit", std::string("s")}});
        auto& sets                       = fixture.emplace<SetSink>();

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, histogram, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.connect<"out", "in">(histogram, sets).has_value());
        expect(fixture.run().has_value());
        expect(ge(sets._sets.size(), 1UZ));

        { // when the snapshot was asked for, on the stream's own event axis
            gr::testing::MarbleDiagram diagram{"Histogram: the event that published, and the figures it carried"};
            diagram.row("evtIn").at(0U, "snapshot");
            diagram.condition(std::format("Histogram(20 bins over [0, 1) s), mean {:.3f}, stddev {:.3f}", histogram.mean.value, histogram.stddev.value));
            diagram.row("out").at(0U, "DataSet");
            diagram.print();
        }

        const gr::DataSet<double>&   last = sets._sets.back();
        const double                 peak = *std::ranges::max_element(last.signal_values);
        gr::graphs::ImChart<128, 24> chart({{last.axis_values[0].front(), last.axis_values[0].back()}, {0., peak}});
        chart.draw<gr::graphs::Style::Bars>(last.axis_values[0], last.signal_values, "counts");
        chart.draw();
        std::println("{} entries, mean {:.4f} s, stddev {:.4f} s, {} under, {} over", //
            histogram.n_entries.value, histogram.mean.value, histogram.stddev.value, histogram.n_underflow.value, histogram.n_overflow.value);
    };
};

int main() { /* not needed for UT */ }
