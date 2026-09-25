#include <boost/ut.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/math/Histogram.hpp>
#include <gnuradio-4.0/math/TensorPeakFinder.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/EventScript.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

using namespace gr::blocks::math;
using gr::testing::ProcessFunction;
using gr::testing::TagSource;
using gr::testing::TensorCollector;
using gr::testing::TensorSource;

namespace {
[[nodiscard]] gr::property_map eventNamed(std::string name) {
    const std::string       nameKey{gr::tag::TRIGGER_NAME.key()};
    const std::string       timeKey{gr::tag::TRIGGER_TIME.key()};
    constexpr std::uint64_t kEventTimeNs = 1'000'000'000U;
    return gr::property_map{{nameKey, std::move(name)}, {timeKey, kEventTimeNs}};
}

[[nodiscard]] double totalOf(const gr::DataSet<double>& set) { return std::accumulate(set.signal_values.begin(), set.signal_values.end(), 0.); }

constexpr gr::Size_t kSamples = 500U;

const auto histogramTests = [] {
    using namespace boost::ut;

    const std::vector<double> fiveBins{0.1, 0.3, 0.5, 0.7, 0.9};

    "each value is counted in the bin that holds it"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<TagSource<double>>({{"n_samples_max", kSamples}, {"values", fiveBins}});
        auto&                     histogram = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 5U}, {"reset_on_publish", false}, {"disconnect_on_done", false}});

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.run().has_value());

        expect(eq(histogram._entriesInRange, kSamples)) << "every sample of the run is inside the range";
        expect(eq(histogram._belowRange, 0U));
        expect(eq(histogram._aboveRange, 0U));
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

        expect(eq(histogram._belowRange, 30U));
        expect(eq(histogram._aboveRange, 30U));
        expect(eq(histogram._entriesInRange, 30U)) << "only the third of the samples that is in range is an entry";
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

        expect(approx(histogram._mean, 0.5, 1e-9)) << "two bins would have put the mean at 0.25 or 0.75";
        expect(approx(histogram._stdDev, 0.2829, 1e-3)) << "the sample deviation of {0.1 .. 0.9}";
        expect(approx(histogram._min, 0.1, 1e-9));
        expect(approx(histogram._max, 0.9, 1e-9));
    };

    "the filter decides which event publishes a snapshot"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TagSource<double>>({{"n_samples_max", kSamples}, {"values", fiveBins}});
        auto&                     script = fixture.emplace<gr::testing::EventScript>();
        script._events                   = {eventNamed("other"), eventNamed("snapshot")};
        auto& histogram                  = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 5U}, {"trigger_name", std::string("snapshot")}});
        auto& sets                       = fixture.emplace<gr::testing::DataSetCollector<double>>();

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, histogram, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.connect<"out", "in">(histogram, sets).has_value());
        expect(fixture.run().has_value());

        expect(eq(histogram._snapshotsPublished, 1U)) << "one of the two events is named by the filter, the other is not";
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
        auto&                     script = fixture.emplace<gr::testing::EventScript>();
        script._events                   = {eventNamed("whatever")};
        auto& histogram                  = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 5U}});
        auto& sets                       = fixture.emplace<gr::testing::DataSetCollector<double>>();

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, histogram, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.connect<"out", "in">(histogram, sets).has_value());
        expect(fixture.run().has_value());

        expect(eq(histogram._snapshotsPublished, 1U));
        expect(eq(sets._sets.size(), 1UZ));
    };

    "a periodic snapshot counts each sample once"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<TagSource<double>>({{"n_samples_max", kSamples}, {"values", fiveBins}});
        auto&                     histogram = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 5U}, {"n_samples", 100U}});
        auto&                     sets      = fixture.emplace<gr::testing::DataSetCollector<double>>();

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.connect<"out", "in">(histogram, sets).has_value());
        expect(fixture.run().has_value());

        expect(ge(sets._sets.size(), 1UZ)) << "500 samples at 100 a snapshot publishes at least once";
        const auto   addTotal  = [](double sum, const gr::DataSet<double>& set) { return sum + totalOf(set); };
        const double published = std::accumulate(sets._sets.begin(), sets._sets.end(), 0., addTotal);
        const double counted   = published + static_cast<double>(histogram._entriesInRange);
        expect(approx(counted, static_cast<double>(kSamples), 1e-9)) << "a reset snapshot covers the samples since the last one: none counted twice, none lost";
    };

    "a cumulative snapshot keeps what the previous one counted"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source    = fixture.emplace<TagSource<double>>({{"n_samples_max", kSamples}, {"values", fiveBins}});
        auto&                     histogram = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 5U}, {"n_samples", 100U}, {"reset_on_publish", false}});
        auto&                     sets      = fixture.emplace<gr::testing::DataSetCollector<double>>();

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.connect<"out", "in">(histogram, sets).has_value());
        expect(fixture.run().has_value());

        expect(ge(sets._sets.size(), 1UZ));
        for (std::size_t i = 1UZ; i < sets._sets.size(); ++i) {
            expect(ge(totalOf(sets._sets[i]), totalOf(sets._sets[i - 1UZ]))) << "a cumulative histogram never loses an entry";
        }
        expect(eq(histogram._entriesInRange, kSamples));
    };

    "a snapshot is due once the timeout has passed since the last one"_test = [] {
        using namespace std::chrono_literals;
        Histogram<double> histogram({{"timeout", 0.5}});
        histogram.settings().init();
        std::ignore = histogram.settings().applyStagedParameters();
        const std::chrono::steady_clock::time_point lastPublish{100s};
        histogram._snapshot.lastPublish = lastPublish;

        expect(!histogram.snapshotDue(lastPublish + 499ms));
        expect(histogram.snapshotDue(lastPublish + 500ms)) << "0.5 s after the last snapshot";
        histogram.timeout = 0.;
        expect(!histogram.snapshotDue(lastPublish + 1h)) << "a timeout of 0 never publishes";
    };

    "a bin count of zero and an inverted range are refused"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     histogram = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 5U}});
        histogram.settings().init();
        std::ignore = histogram.settings().applyStagedParameters();
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
        expect(eq(histogram._histogram.entriesInRange, std::uint64_t{10}));

        expect(histogram.settings().set({{"axis_unit", gr::Tensor<gr::pmt::Value>(gr::data_from, {gr::pmt::Value("ms")})}}).empty());
        std::ignore = histogram.settings().activateContext();
        std::ignore = histogram.settings().applyStagedParameters();
        expect(eq(histogram._histogram.entriesInRange, std::uint64_t{10})) << "settingsChanged fires on any setting, and must not throw an accumulation away";

        expect(histogram.settings().set({{"n_bins", gr::Size_t(10)}}).empty());
        std::ignore = histogram.settings().activateContext();
        std::ignore = histogram.settings().applyStagedParameters();
        expect(eq(histogram._histogram.entriesInRange, std::uint64_t{0})) << "a change of shape has no counts to keep";
        expect(eq(histogram._counts.size(), 10UZ));
        expect(eq(histogram._histogram.bins.size(), 10UZ)) << "and the accumulator views the new storage, not the old";
    };

    "the distribution a run produced, drawn"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TagSource<double>>({{"n_samples_max", 1000U}, {"values", std::vector<double>{0.35, 0.45, 0.45, 0.5, 0.5, 0.5, 0.55, 0.55, 0.65, 0.5}}});
        auto&                     script = fixture.emplace<gr::testing::EventScript>();
        script._events                   = {eventNamed("snapshot")};
        const gr::Tensor<gr::pmt::Value> axisName(gr::data_from, {gr::pmt::Value("interval")});
        const gr::Tensor<gr::pmt::Value> axisUnit(gr::data_from, {gr::pmt::Value("s")});
        auto&                            histogram = fixture.emplace<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1.}, {"n_bins", 20U}, {"trigger_name", std::string("snapshot")}, {"reset_on_publish", false}, {"axis_name", axisName}, {"axis_unit", axisUnit}});
        auto&                            sets      = fixture.emplace<gr::testing::DataSetCollector<double>>();

        expect(fixture.connect<"out", "in">(source, histogram).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, histogram, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.connect<"out", "in">(histogram, sets).has_value());
        expect(fixture.run().has_value());
        expect(ge(sets._sets.size(), 1UZ));

        {
            gr::testing::MarbleDiagram diagram{"Histogram: the event that published, and the figures it carried"};
            diagram.row("evtIn").at(0U, "snapshot");
            diagram.condition(std::format("Histogram(20 bins over [0, 1) s), mean {:.3f}, stddev {:.3f}", histogram._mean, histogram._stdDev));
            diagram.row("out").at(0U, "DataSet");
            diagram.print();
        }

        const gr::DataSet<double>& last = sets._sets.back();
        expect(eq(last.axis_names.at(0UZ), std::string_view("interval"))) << "the axis name setting reaches the published DataSet";
        expect(eq(last.axis_units.at(0UZ), std::string_view("s")));
        const double                 peak = *std::ranges::max_element(last.signal_values);
        gr::graphs::ImChart<128, 24> chart({{last.axis_values[0].front(), last.axis_values[0].back()}, {0., peak}});
        chart.draw<gr::graphs::Style::Bars>(last.axis_values[0], last.signal_values, "counts");
        chart.draw();
        std::println("{} entries, mean {:.4f} s, stddev {:.4f} s, {} under, {} over", //
            histogram._entriesInRange, histogram._mean, histogram._stdDev, histogram._belowRange, histogram._aboveRange);
    };
};

namespace histogram2D {
using namespace boost::ut;

struct Binned {
    std::vector<double>                                   bins;
    double                                                entriesInRange = 0.;
    double                                                outsideRange   = 0.;
    gr::Size_t                                            snapshots      = 0U;
    std::vector<std::vector<double>>                      maps;
    std::vector<std::vector<std::size_t>>                 shapes;
    std::vector<std::pair<std::size_t, gr::property_map>> tags;
    bool                                                  ran = false;
};

template<typename T>
[[nodiscard]] Binned collect(gr::testing::GraphFixture<>& fixture, Histogram2D<T>& histogram) {
    auto& sink = fixture.emplace<TensorCollector<T>>();
    expect(fixture.connect<"out", "in">(histogram, sink).has_value());
    const bool ran = fixture.run().has_value();
    Binned     binned{.bins = histogram._bins, .entriesInRange = histogram._entriesInRange, .outsideRange = histogram._outsideRange, .snapshots = histogram._snapshotsPublished, .maps = {}, .shapes = sink._shapes, .tags = sink._tags, .ran = ran};
    for (const auto& elements : sink._elements) {
        binned.maps.emplace_back(elements.begin(), elements.end());
    }
    return binned;
}

[[nodiscard]] Binned binPairs(std::vector<double> xs, std::vector<double> ys, gr::Size_t nPairs, gr::property_map settings, std::vector<gr::property_map> events = {}) {
    gr::testing::GraphFixture fixture;
    auto&                     xSource   = fixture.emplace<TagSource<double>>({{"n_samples_max", nPairs}, {"values", std::move(xs)}});
    auto&                     ySource   = fixture.emplace<TagSource<double>>({{"n_samples_max", nPairs}, {"values", std::move(ys)}});
    auto*                     script    = events.empty() ? nullptr : &fixture.emplace<gr::testing::EventScript>();
    auto&                     histogram = fixture.emplace<Histogram2D<double>>(std::move(settings));
    expect(fixture.connect<"out", "x">(xSource, histogram).has_value());
    expect(fixture.connect<"out", "y">(ySource, histogram).has_value());
    if (script != nullptr) {
        script->_events = std::move(events);
        expect(fixture.graph.connect(*script, gr::PortDefinition{std::string("evtOut")}, histogram, gr::PortDefinition{std::string("evtIn")}).has_value());
    }
    return collect(fixture, histogram);
}

template<typename T>
[[nodiscard]] Binned binPoints(std::vector<typename TensorSource<T>::Frame> frames, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<TensorSource<T>>();
    source._frames                   = std::move(frames);
    auto& histogram                  = fixture.emplace<Histogram2D<T>>(std::move(settings));
    expect(fixture.connect<"out", "points">(source, histogram).has_value());
    return collect(fixture, histogram);
}

[[nodiscard]] double totalOf(const std::vector<double>& map) { return std::accumulate(map.begin(), map.end(), 0.); }

void expectBins(const Binned& binned, const std::vector<double>& expected, std::string_view context) {
    expect(fatal(eq(binned.bins.size(), expected.size()))) << context;
    for (std::size_t bin = 0UZ; bin < expected.size(); ++bin) {
        expect(approx(binned.bins[bin], expected[bin], 1e-12)) << context << "bin" << bin;
    }
}

[[nodiscard]] std::vector<double> tensorOf(const gr::property_map& tag, std::string_view key) {
    const auto values = tag.value_or<gr::Tensor<double>>(key, gr::Tensor<double>{});
    return {values.begin(), values.end()};
}

[[nodiscard]] std::vector<std::string> labelsOf(const gr::property_map& tag, std::string_view key) {
    const auto               values = tag.value_or<gr::Tensor<gr::pmt::Value>>(key, gr::Tensor<gr::pmt::Value>{});
    std::vector<std::string> labels;
    for (const gr::pmt::Value& value : values) {
        labels.emplace_back(value.value_or(std::string_view{}));
    }
    return labels;
}

const gr::property_map kUnitSquare2x2{{"n_bins_x", 2U}, {"n_bins_y", 2U}, {"x_min", 0.}, {"x_max", 1.}, {"y_min", 0.}, {"y_max", 1.}, {"reset_on_publish", false}};

const auto histogram2DTests = [] {
    "each pair from x and y is counted in the bin that holds it, x along a row"_test = [] {
        const Binned binned = binPairs({0.1, 0.6, 0.6, 0.1, 0.35}, {0.1, 0.1, 0.6, 0.6, 0.85}, 10U, kUnitSquare2x2);
        expect(binned.ran);
        expectBins(binned, {2., 2., 4., 2.}, "(0.1,0.1) row 0 column 0, (0.6,0.1) row 0 column 1, (0.6,0.6) row 1 column 1, (0.1,0.6) and (0.35,0.85) row 1 column 0, twice each");
        expect(eq(binned.entriesInRange, 10.));
        expect(eq(binned.outsideRange, 0.));
    };

    "a pair outside the range or with a NaN is counted apart rather than clamped into an edge bin"_test = [] {
        const double nan    = std::numeric_limits<double>::quiet_NaN();
        const Binned binned = binPairs({-0.5, 0.5, 1.5, 0.5, 0.5}, {0.5, 0.5, 0.5, nan, 1.0}, 100U, {{"n_bins_x", 4U}, {"n_bins_y", 1U}, {"reset_on_publish", false}});
        expect(binned.ran);
        expectBins(binned, {0., 0., 20., 0.}, "only (0.5,0.5) is inside: x left and right of the range, y NaN and y on its excluded upper edge are not");
        expect(eq(binned.entriesInRange, 20.));
        expect(eq(binned.outsideRange, 80.));
    };

    "rows of a points tensor are binned, each counting 1 without a weight column"_test = [] {
        const Binned binned = binPoints<double>({{.extents = {3UZ, 2UZ}, .elements = {0.1, 0.1, 0.6, 0.6, 0.6, 0.6}}, {.extents = {1UZ, 2UZ}, .elements = {0.1, 0.6}}}, kUnitSquare2x2);
        expect(binned.ran);
        expectBins(binned, {1., 0., 1., 2.}, "(0.1,0.1) once, (0.1,0.6) once, (0.6,0.6) twice, across two frames");
        expect(eq(binned.entriesInRange, 4.));
    };

    "the weight column weighs each row, the columns beyond it ignored"_test = [] {
        gr::property_map settings = kUnitSquare2x2;
        settings.insert_or_assign("weight_column", 2);
        const Binned binned = binPoints<float>({{.extents = {2UZ, 4UZ}, .elements = {0.1f, 0.1f, 0.5f, 9.f, 0.6f, 0.6f, 2.f, 9.f}}}, settings);
        expect(binned.ran);
        expectBins(binned, {0.5, 0., 0., 2.}, "(0.1,0.1) weighs 0.5, (0.6,0.6) weighs 2");
        expect(eq(binned.entriesInRange, 2.5));
    };

    "a points frame without the weight column is refused and nothing of it is binned"_test = [] {
        gr::property_map settings = kUnitSquare2x2;
        settings.insert_or_assign("weight_column", 2);
        const Binned binned = binPoints<double>({{.extents = {1UZ, 3UZ}, .elements = {0.1, 0.1, 0.5}}, {.extents = {1UZ, 2UZ}, .elements = {0.1, 0.1}}, {.extents = {1UZ, 3UZ}, .elements = {0.6, 0.6, 2.}}}, settings);
        expect(!binned.ran) << "the refusal stops the graph with an error";
        expect(eq(binned.entriesInRange, 0.5)) << "the second frame has columns 0 and 1 only: binning stops there, and the third frame is not reached";
        expect(eq(binned.snapshots, 0U));
    };

    "x without y is refused: pairs need both"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     xSource   = fixture.emplace<TagSource<double>>({{"n_samples_max", 10U}, {"values", std::vector<double>{0.1}}});
        auto&                     histogram = fixture.emplace<Histogram2D<double>>(kUnitSquare2x2);
        expect(fixture.connect<"out", "x">(xSource, histogram).has_value());
        const Binned binned = collect(fixture, histogram);
        expect(!binned.ran) << "the refusal stops the graph with an error, where a silently ignored x would end cleanly";
        expect(eq(binned.entriesInRange + binned.outsideRange, 0.));
    };

    "the map forgets by exp(-dt / time_constant), independent of how dt is split"_test = [] {
        using namespace std::chrono_literals;
        const auto agedBy = [](double timeConstant, const std::vector<std::chrono::milliseconds>& steps) {
            Histogram2D<double> histogram(kUnitSquare2x2);
            histogram.settings().init();
            std::ignore             = histogram.settings().applyStagedParameters();
            histogram.time_constant = timeConstant;
            histogram.start();
            histogram._histogram.add(0.1, 0.1, 4.);
            histogram._histogram.add(2., 2., 1.);
            const std::chrono::steady_clock::time_point start{100s};
            histogram._lastAged = start;
            std::chrono::milliseconds elapsed{0};
            for (const std::chrono::milliseconds step : steps) {
                elapsed += step;
                histogram.age(start + elapsed);
            }
            return std::pair{histogram._bins[0], histogram._histogram.outsideRange};
        };
        const auto [once, onceOutside] = agedBy(2., {2000ms});
        expect(approx(once, 4. * std::exp(-1.), 1e-12)) << "after one time constant a bin is 1/e of what it was";
        expect(approx(onceOutside, std::exp(-1.), 1e-12)) << "the outside sum ages with it";
        const auto [inSteps, inStepsOutside] = agedBy(2., {500ms, 500ms, 1000ms});
        expect(approx(inSteps, once, 1e-12)) << "three work calls over the same 2 s forget as much as one";
        const auto [forever, foreverOutside] = agedBy(0., {3'600'000ms});
        expect(eq(forever, 4.)) << "a time constant of 0 accumulates forever";
        expect(eq(foreverOutside, 1.));
    };

    "a snapshot is due once the timeout has passed since the last one"_test = [] {
        using namespace std::chrono_literals;
        Histogram2D<double> histogram({{"timeout", 0.5}});
        histogram.settings().init();
        std::ignore = histogram.settings().applyStagedParameters();
        const std::chrono::steady_clock::time_point lastPublish{100s};
        histogram._snapshot.lastPublish = lastPublish;

        expect(!histogram.snapshotDue(lastPublish + 499ms));
        expect(histogram.snapshotDue(lastPublish + 500ms)) << "0.5 s after the last snapshot";
        histogram.timeout = 0.;
        expect(!histogram.snapshotDue(lastPublish + 1h)) << "a timeout of 0 never publishes";
    };

    "a periodic snapshot counts each pair once"_test = [] {
        gr::property_map settings = kUnitSquare2x2;
        settings.insert_or_assign("n_samples", 100U);
        settings.insert_or_assign("reset_on_publish", true);
        const Binned binned = binPairs({0.1, 0.6}, {0.6, 0.1}, 500U, settings);
        expect(binned.ran);
        expect(ge(binned.maps.size(), 1UZ)) << "500 pairs at 100 a snapshot publish at least once";
        const auto   addTotal  = [](double sum, const std::vector<double>& map) { return sum + totalOf(map); };
        const double published = std::accumulate(binned.maps.begin(), binned.maps.end(), 0., addTotal);
        expect(approx(published + binned.entriesInRange, 500., 1e-9)) << "a reset snapshot covers the pairs since the last one: none counted twice, none lost";
        for (const std::vector<std::size_t>& shape : binned.shapes) {
            expect(shape == std::vector<std::size_t>{2UZ, 2UZ}) << "each map is [n_bins_y, n_bins_x]";
        }
    };

    "a cumulative snapshot keeps what the previous one counted"_test = [] {
        gr::property_map settings = kUnitSquare2x2;
        settings.insert_or_assign("n_samples", 100U);
        const Binned binned = binPairs({0.1, 0.6}, {0.6, 0.1}, 500U, settings);
        expect(ge(binned.maps.size(), 1UZ));
        for (std::size_t i = 1UZ; i < binned.maps.size(); ++i) {
            expect(ge(totalOf(binned.maps[i]), totalOf(binned.maps[i - 1UZ]))) << "a cumulative map never loses an entry";
        }
        expect(eq(binned.entriesInRange, 500.));
    };

    "the filter decides which event publishes a snapshot, and a reset one starts the next empty"_test = [] {
        gr::property_map settings = kUnitSquare2x2;
        settings.insert_or_assign("trigger_name", std::string("snapshot"));
        settings.insert_or_assign("reset_on_publish", true);
        const Binned binned = binPairs({0.1, 0.6}, {0.6, 0.1}, 500U, settings, {eventNamed("other"), eventNamed("snapshot")});
        expect(eq(binned.snapshots, 1U)) << "one of the two events is named by the filter, the other is not";
        expect(fatal(eq(binned.maps.size(), 1UZ)));
        expect(approx(totalOf(binned.maps[0]) + binned.entriesInRange, 500., 1e-9)) << "what the event published is no longer counted";
    };

    "the axes of the first map are its bin centres and widths, y first"_test = [] {
        const gr::Tensor<gr::pmt::Value> axisName(gr::data_from, {gr::pmt::Value("delay"), gr::pmt::Value("amplitude")});
        const gr::Tensor<gr::pmt::Value> axisUnit(gr::data_from, {gr::pmt::Value("s"), gr::pmt::Value("V")});
        const gr::property_map           settings{{"n_bins_x", 4U}, {"n_bins_y", 5U}, {"x_min", -1.}, {"x_max", 1.}, {"y_min", 10.}, {"y_max", 20.}, {"n_samples", 40U}, {"axis_name", axisName}, {"axis_unit", axisUnit}};
        const Binned                     binned = binPairs({0.}, {15.}, 40U, settings);
        expect(fatal(ge(binned.maps.size(), 1UZ)));
        expect(binned.shapes[0] == std::vector<std::size_t>{5UZ, 4UZ});
        expect(fatal(ge(binned.tags.size(), 1UZ))) << "the axes go out with the first map";
        const auto& [index, axes] = binned.tags[0];
        expect(eq(index, 0UZ));
        expect(tensorOf(axes, gr::tag::AXIS_ORIGIN.key()) == std::vector<double>{11., -0.75}) << "the centre of bin 0: 10 + 2/2 in y, -1 + 0.5/2 in x";
        expect(tensorOf(axes, gr::tag::AXIS_STEP.key()) == std::vector<double>{2., 0.5}) << "10 over 5 rows, 2 over 4 columns";
        expect(labelsOf(axes, gr::tag::AXIS_NAME.key()) == std::vector<std::string>{"delay", "amplitude"});
        expect(labelsOf(axes, gr::tag::AXIS_UNIT.key()) == std::vector<std::string>{"s", "V"});
        expect(eq(binned.maps[0][2UZ * 4UZ + 2UZ], 40.)) << "(0, 15) is row 2, column 2";
    };

    "a bin count of zero, an inverted range, a weight column on x or y and a negative time constant are refused"_test = [] {
        Histogram2D<double> histogram(kUnitSquare2x2);
        histogram.settings().init();
        std::ignore = histogram.settings().applyStagedParameters();

        expect(histogram.settings().set({{"n_bins_x", gr::Size_t(0)}, {"y_max", -1.}, {"weight_column", 1}, {"time_constant", -1.}}).empty());
        std::ignore = histogram.settings().activateContext();
        std::ignore = histogram.settings().applyStagedParameters();

        expect(eq(histogram.n_bins_x.value, 2U)) << "a map of no columns counts nothing, so the previous count stays";
        expect(approx(histogram.y_max.value, 1., 1e-12)) << "an inverted range has no bin to fill";
        expect(eq(histogram.weight_column.value, -1)) << "column 1 is y";
        expect(eq(histogram.time_constant.value, 0.)) << "a negative time constant would grow the map";
    };

    "a setting that leaves the bins alone keeps what was counted"_test = [] {
        Histogram2D<double> histogram(kUnitSquare2x2);
        histogram.settings().init();
        std::ignore = histogram.settings().applyStagedParameters();
        histogram.start();
        histogram._histogram.add(0.5, 0.5);
        expect(histogram.settings().set({{"time_constant", 5.}}).empty());
        std::ignore = histogram.settings().activateContext();
        std::ignore = histogram.settings().applyStagedParameters();
        expect(eq(histogram._histogram.entriesInRange, 1.)) << "settingsChanged fires on any setting, and must not throw an accumulation away";

        expect(histogram.settings().set({{"n_bins_y", gr::Size_t(3)}}).empty());
        std::ignore = histogram.settings().activateContext();
        std::ignore = histogram.settings().applyStagedParameters();
        expect(eq(histogram._histogram.entriesInRange, 0.)) << "a change of shape has no counts to keep";
        expect(eq(histogram._histogram.bins.size(), 6UZ)) << "and the accumulator views the new storage, not the old";
    };

    "TensorPeakFinder downstream finds a cluster at the centre of its bin, in the units of x and y"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     xSource   = fixture.emplace<TagSource<double>>({{"n_samples_max", 400U}, {"values", std::vector<double>{0.65, 0.65, 0.65, 0.15}}});
        auto&                     ySource   = fixture.emplace<TagSource<double>>({{"n_samples_max", 400U}, {"values", std::vector<double>{0.35, 0.35, 0.35, 0.85}}});
        auto&                     histogram = fixture.emplace<Histogram2D<double>>({{"n_bins_x", 10U}, {"n_bins_y", 10U}, {"n_samples", 400U}});
        auto&                     finder    = fixture.emplace<TensorPeakFinder<double>>({{"max_peaks", 2U}, {"threshold", 0.5f}});
        auto&                     sink      = fixture.emplace<TensorCollector<double>>();
        expect(fixture.connect<"out", "x">(xSource, histogram).has_value());
        expect(fixture.connect<"out", "y">(ySource, histogram).has_value());
        expect(fixture.connect<"out", "in">(histogram, finder).has_value());
        expect(fixture.connect<"out", "in">(finder, sink).has_value());
        expect(fixture.run().has_value());

        expect(fatal(eq(sink._shapes.size(), 1UZ))) << "400 pairs at 400 a snapshot publish once";
        expect(sink._shapes[0] == std::vector<std::size_t>{1UZ, 4UZ}) << "one peak {y, x, score, confidence}: the cluster of 100 is below half of the one of 300";
        expect(approx(sink._elements[0][0], 0.35, 1e-12)) << "y at the centre of row 3";
        expect(approx(sink._elements[0][1], 0.65, 1e-12)) << "x at the centre of column 6";
        expect(approx(sink._elements[0][2], 300., 1e-9)) << "the score is the bin content, its neighbours empty";
    };
};
} // namespace histogram2D

} // namespace

int main() {
    using namespace boost::ut;
    "Histogram"_test   = histogramTests;
    "Histogram2D"_test = histogram2D::histogram2DTests;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
