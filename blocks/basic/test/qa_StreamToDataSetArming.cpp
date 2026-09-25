#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/basic/StreamToDataSet.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/EventScript.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

using namespace gr::basic;
using gr::testing::ProcessFunction;
using gr::testing::TagSource;

namespace {
struct TaggedRamp : gr::Block<TaggedRamp> {
    gr::PortOut<float> out;

    GR_MAKE_REFLECTABLE(TaggedRamp, out);

    std::vector<std::pair<std::size_t, gr::property_map>> _tags;
    std::size_t                                           _nSamples = 0UZ;
    std::size_t                                           _produced = 0UZ;

    gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        std::size_t emitted = 0UZ;
        while (emitted < outSpan.size() && _produced < _nSamples) {
            for (const auto& [at, map] : _tags) {
                if (at == _produced) {
                    outSpan.publishTag(map, emitted);
                }
            }
            outSpan[emitted] = static_cast<float>(_produced);
            ++emitted;
            ++_produced;
        }
        outSpan.publish(emitted);
        if (_produced >= _nSamples) {
            this->requestStop();
            return gr::work::Status::DONE;
        }
        return gr::work::Status::OK;
    }
};

struct ManualRig {
    StreamToDataSet<float>         extract;
    gr::PortOut<float>             source;
    gr::EventPortOut               events;
    gr::PortIn<gr::DataSet<float>> sink;

    explicit ManualRig(gr::property_map settings) : extract(std::move(settings)) {
        boost::ut::expect(source.connect(extract.in).has_value()) << boost::ut::fatal;
        boost::ut::expect(events.connect(extract.evtIn).has_value()) << boost::ut::fatal;
        boost::ut::expect(extract.out.connect(sink).has_value()) << boost::ut::fatal;
        extract.init(extract.progress);
        boost::ut::expect(extract.changeStateTo(gr::lifecycle::State::RUNNING).has_value()) << boost::ut::fatal;
    }
};

[[nodiscard]] gr::property_map named(std::string name) { return gr::property_map{{gr::testing::kTriggerKey, std::move(name)}}; }

struct Outcome {
    std::size_t sets     = 0UZ;
    gr::Size_t  captured = 0U;
    gr::Size_t  refused  = 0U;
    gr::Size_t  busy     = 0U;
};

[[nodiscard]] Outcome capture(std::vector<std::pair<std::size_t, gr::property_map>> tags, std::size_t nSamples, gr::property_map settings) {
    settings[std::string("filter")] = std::string("start");
    settings[std::string("n_post")] = gr::Size_t(2);
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<TaggedRamp>();
    source._tags                     = std::move(tags);
    source._nSamples                 = nSamples;
    auto& extract                    = fixture.emplace<StreamToDataSet<float>>(std::move(settings));
    auto& sink                       = fixture.emplace<gr::testing::DataSetCollector<float>>();

    boost::ut::expect(fixture.connect<"out", "in">(source, extract).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(extract, sink).has_value());
    boost::ut::expect(fixture.run().has_value());

    return Outcome{.sets = sink._sets.size(), .captured = extract._segmentsCaptured, .refused = extract._startsRefused, .busy = extract._startsWhileBusy};
}
const auto arming = [] {
    using namespace boost::ut;

    const std::vector<std::pair<std::size_t, gr::property_map>> threeStarts{{2UZ, named("start")}, {8UZ, named("start")}, {14UZ, named("start")}};

    "without arming every trigger opens a range, as before"_test = [&] {
        const auto found = capture(threeStarts, 20UZ, {});

        expect(eq(found.captured, 3U));
        expect(eq(found.refused, 0U));
        expect(eq(found.sets, 3UZ));
    };

    "with an arming filter nothing is captured until the arming trigger arrives"_test = [&] {
        std::vector<std::pair<std::size_t, gr::property_map>> tags = threeStarts;
        tags.emplace_back(10UZ, named("arm"));

        const auto found = capture(std::move(tags), 20UZ, {{"arm_filter", std::string("arm")}});

        expect(eq(found.captured, 1U)) << "only the start that followed the arming trigger";
        expect(eq(found.refused, 2U)) << "and the two before it are counted";
    };

    "manual rearming allows one range per arming trigger"_test = [&] {
        std::vector<std::pair<std::size_t, gr::property_map>> tags = threeStarts;
        tags.emplace_back(0UZ, named("arm"));

        const auto found = capture(std::move(tags), 20UZ, {{"arm_filter", std::string("arm")}, {"rearm", std::string("manual")}});

        expect(eq(found.captured, 1U));
        expect(eq(found.refused, 2U)) << "the arming was spent on the first range";
    };

    "a segment limit stops the block opening more"_test = [&] {
        const auto found = capture(threeStarts, 20UZ, {{"n_segments", 2U}});

        expect(eq(found.captured, 2U));
        expect(eq(found.refused, 1U));
        expect(eq(found.sets, 2UZ));
    };

    "a holdoff makes the block deaf for a while after a range opens"_test = [&] {
        const auto found = capture(threeStarts, 20UZ, {{"holdoff_samples", 8U}});

        expect(eq(found.captured, 2U)) << "the start at 8 falls inside the holdoff of the one at 2";
        expect(eq(found.refused, 1U));
    };

    "a holdoff in seconds is the same thing where the rate is known"_test = [&] {
        const auto found = capture(threeStarts, 20UZ, {{"holdoff_seconds", 0.008f}, {"sample_rate", 1000.f}});

        expect(eq(found.captured, 2U)) << "8 ms at 1 kHz is eight samples";
        expect(eq(found.refused, 1U));
    };

    "an injected event opens a range in a stream that carries no tags"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TaggedRamp>();
        source._nSamples                 = 40UZ;
        auto& script                     = fixture.emplace<gr::testing::EventScript>();
        script._events                   = {named("start")};
        auto& extract                    = fixture.emplace<StreamToDataSet<float>>({{"filter", std::string("start")}, {"n_post", 2U}});
        auto& sink                       = fixture.emplace<gr::testing::DataSetCollector<float>>();

        expect(fixture.connect<"out", "in">(source, extract).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, extract, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.connect<"out", "in">(extract, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(extract._segmentsCaptured, 1U)) << "the event stood in for the tag the stream never carried";
        expect(ge(sink._sets.size(), 1UZ));
    };

    "what the gate let through, drawn"_test = [&] {
        const auto found = capture(threeStarts, 20UZ, {{"holdoff_samples", 8U}});

        gr::testing::MarbleDiagram diagram{"StreamToDataSet: a holdoff of 8 samples after each range opens"};
        diagram.unit = "sample";
        diagram.row("in").at(2U, "start").at(8U, "start").at(14U, "start").completes();
        diagram.condition(std::format("StreamToDataSet(filter = \"start\", holdoff = 8 samples), {} refused", found.refused));
        diagram.row("out").at(2U, "DataSet").at(14U, "DataSet").completes();
        diagram.print();

        expect(eq(found.captured, 2U));
    };

    "a work call without stream samples keeps an open single-trigger range open"_test = [] {
        ManualRig rig({{"filter", std::string("start")}, {"n_post", gr::Size_t(10)}});
        auto& [extract, source, events, sink] = rig;

        const auto publishSamples = [&source](std::size_t nSamples) {
            auto samples = source.streamWriter().reserve<gr::SpanReleasePolicy::ProcessAll>(nSamples);
            std::ranges::fill(samples, 1.f);
            samples.publish(nSamples);
        };

        source.publishTag(gr::property_map_view{named("start")}, 0UZ);
        publishSamples(4UZ);
        std::ignore = extract.work(4UZ);
        {
            auto span = events.reserve<gr::SpanReleasePolicy::ProcessNone>(1UZ);
            expect(gr::emit(span, 0UZ, gr::property_map_view{named("unrelated")}).has_value()) << fatal;
            span.publish(1UZ);
        }
        expect(eq(extract.in.streamReader().available(), 0UZ));
        std::ignore = extract.work(1UZ);
        publishSamples(6UZ);
        std::ignore = extract.work(6UZ);

        auto sets = sink.streamReader().get<gr::SpanReleasePolicy::ProcessAll>(sink.streamReader().available());
        expect(eq(sets.size(), 1UZ)) << fatal;
        expect(eq(sets[0].signal_values.size(), 10UZ)) << "an empty call must not close the range before n_post samples";
    };

    "a start while a stream range is open is busy and does not use up a segment"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TaggedRamp>();
        source._tags                     = {{2UZ, named("start")}, {4UZ, named("start")}, {14UZ, named("start")}};
        source._nSamples                 = 20UZ;
        auto& extract                    = fixture.emplace<StreamFilter<float>>({{"filter", std::string("start")}, {"n_post", gr::Size_t(5)}, {"n_segments", gr::Size_t(2)}});
        auto& sink                       = fixture.emplace<gr::testing::TagSink<float, ProcessFunction::USE_PROCESS_BULK>>();

        expect(fixture.connect<"out", "in">(source, extract).has_value());
        expect(fixture.connect<"out", "in">(extract, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(extract._startsWhileBusy, 1U)) << "the start at 4 arrives inside the range opened at 2";
        expect(eq(extract._segmentsCaptured, 2U));
        expect(eq(extract._startsRefused, 0U)) << "the busy start must not spend the second segment";
        expect(eq(sink._samples.size(), 10UZ)) << "two ranges of n_post samples";
    };

    "a reset starts the counters, and with them the segment limit, afresh"_test = [] {
        ManualRig rig({{"filter", std::string("start")}, {"n_post", gr::Size_t(2)}, {"n_segments", gr::Size_t(1)}});
        auto& [extract, source, events, sink] = rig;

        const auto captureOnce = [&source, &extract] {
            source.publishTag(gr::property_map_view{named("start")}, 0UZ);
            {
                auto samples = source.streamWriter().reserve<gr::SpanReleasePolicy::ProcessAll>(4UZ);
                std::ranges::fill(samples, 1.f);
                samples.publish(4UZ);
            }
            std::ignore = extract.work(4UZ);
        };

        captureOnce();
        expect(eq(extract._segmentsCaptured, 1U)) << fatal;
        expect(extract.changeStateTo(gr::lifecycle::State::REQUESTED_STOP).has_value()) << fatal;
        expect(extract.changeStateTo(gr::lifecycle::State::STOPPED).has_value()) << fatal;
        expect(extract.changeStateTo(gr::lifecycle::State::INITIALISED).has_value()) << fatal;
        expect(eq(extract._segmentsCaptured, 0U));
        expect(eq(extract._startsRefused, 0U));
        expect(eq(extract._startsWhileBusy, 0U));
        expect(extract.changeStateTo(gr::lifecycle::State::RUNNING).has_value()) << fatal;

        captureOnce();
        expect(eq(extract._segmentsCaptured, 1U)) << "the segment limit counts from the reset";
        expect(eq(extract._startsRefused, 0U));
        expect(eq(sink.streamReader().available(), 2UZ)) << "one DataSet before the reset, one after";
    };

    "a rearm mode other than auto or manual is refused and the previous one kept"_test = [] {
        StreamToDataSet<float> extract({{"filter", std::string("start")}, {"rearm", std::string("manual")}});
        extract.settings().init();
        std::ignore = extract.settings().applyStagedParameters();
        expect(eq(extract.rearm.value, std::string("manual"))) << fatal;

        expect(extract.settings().set({{"rearm", std::string("sometimes")}}).empty());
        std::ignore = extract.settings().activateContext();
        std::ignore = extract.settings().applyStagedParameters();

        expect(eq(extract.rearm.value, std::string("manual"))) << "an unknown mode must not fall back to auto in silence";
    };
};

} // namespace

int main() {
    using namespace boost::ut;
    "StreamToDataSet arming, holdoff and segment limit"_test = arming;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
