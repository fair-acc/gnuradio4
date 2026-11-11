#include <boost/ut.hpp>

#include <algorithm>
#include <cstdint>
#include <format>
#include <numeric>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
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
    std::size_t              sets     = 0UZ;
    gr::Size_t               captured = 0U;
    gr::Size_t               refused  = 0U;
    gr::Size_t               busy     = 0U;
    std::vector<std::string> origins; /// per set, the acquisition flag that says why it opened without a start trigger, else ""
    std::vector<float>       firsts;  /// per set, its first sample, which the ramp makes its stream index
    std::string              stage;
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

    Outcome found{.sets = sink._sets.size(), .captured = extract._segmentsCaptured, .refused = extract._startsRefused, .busy = extract._startsWhileBusy, .origins = {}, .firsts = {}, .stage = extract.stage.value};
    for (const gr::DataSet<float>& set : sink._sets) {
        std::string origin;
        if (!set.meta_information.empty()) {
            if (const auto meta = gr::property_map_view{set.meta_information[0]}.get_if<gr::property_map>(gr::tag::kTriggerMetaInfo.key())) {
                if (const auto flags = gr::property_map_view{*meta}.get_if<gr::property_map>(gr::tag::kAcquisitionFlags.key())) {
                    for (const auto& key : flags->keys()) {
                        origin = std::string(key);
                    }
                }
            }
        }
        found.origins.push_back(origin);
        found.firsts.push_back(set.signal_values.empty() ? -1.f : set.signal_values.front());
    }
    return found;
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

    "single opens one range and then stops, as the recorder's single mode does"_test = [&] {
        const auto found = capture(threeStarts, 20UZ, {{"mode", std::string("single")}});
        expect(eq(found.captured, 1U));
        expect(eq(found.refused, 2U)) << "the later starts are seen and turned down";
        expect(eq(found.stage, std::string("stopped")));
    };

    "a count holdoff refuses the next n starts after each admitted one"_test = [&] {
        const auto found = capture(threeStarts, 20UZ, {{"holdoff_count", 1U}});
        expect(eq(found.captured, 2U)) << "the first and the third";
        expect(eq(found.refused, 1U));
    };

    "a forced trigger opens a range the filter would not, flagged forced"_test = [&] {
        const auto found = capture({{2UZ, named("other")}, {8UZ, named("forced")}}, 20UZ, {});
        expect(eq(found.captured, 1U));
        expect(found.origins == std::vector<std::string>{"forced"});
        expect(found.firsts == std::vector<float>{8.f}) << "the range starts at the forcing tag";
    };

    "auto forces a range at each deadline without a start, flagged auto_timeout"_test = [&] {
        const auto found = capture({}, 20UZ, {{"mode", std::string("auto")}, {"auto_timeout", 5e-3f}, {"sample_rate", 1000.f}});
        expect(found.firsts == std::vector<float>{5.f, 10.f, 15.f}) << "every 5 samples at 1 kHz, from the start and then from each forced range's start" << std::format("{}", found.firsts);
        expect(found.origins == std::vector<std::string>{"auto_timeout", "auto_timeout", "auto_timeout"}) << "none of them is evidence that a trigger matched";
    };

    "a real start on auto's deadline is a real start, not a forced one"_test = [&] {
        const auto found = capture({{5UZ, named("start")}}, 12UZ, {{"mode", std::string("auto")}, {"auto_timeout", 5e-3f}, {"sample_rate", 1000.f}});
        expect(!found.origins.empty() && found.origins.front().empty()) << "the range at 5 was opened by its tag";
    };

    "a forced capture works after single mode has stopped, and does not use up its range"_test = [&] {
        const auto found = capture({{2UZ, named("start")}, {8UZ, named("forced")}, {14UZ, named("start")}}, 20UZ, {{"mode", std::string("single")}});
        expect(eq(found.captured, 2U)) << "the single range and the forced one";
        expect(found.origins == std::vector<std::string>{"", "forced"});
    };

    "a stream filter forces its ranges at the same deadlines"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TaggedRamp>();
        source._nSamples                 = 20UZ;
        auto& filter                     = fixture.emplace<StreamFilter<float>>({{"filter", std::string("start")}, {"n_post", gr::Size_t(2)}, {"mode", std::string("auto")}, {"auto_timeout", 5e-3f}, {"sample_rate", 1000.f}});
        auto& sink                       = fixture.emplace<gr::testing::TagSink<float, ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.connect<"out", "in">(source, filter).has_value());
        expect(fixture.connect<"out", "in">(filter, sink).has_value());
        expect(fixture.run().has_value());
        expect(sink._samples == std::vector<float>{5.f, 6.f, 10.f, 11.f, 15.f, 16.f}) << "two samples (n_post) from each deadline";
    };

    "the range holds the indices [anchor - n_pre, anchor + n_post) exactly"_test = [] {
        const auto found = capture({{10UZ, named("start")}}, 20UZ, {{"n_pre", gr::Size_t(3)}});
        expect(eq(found.sets, 1UZ)) << fatal;
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TaggedRamp>();
        source._tags                     = {{10UZ, named("start")}};
        source._nSamples                 = 20UZ;
        auto& extract                    = fixture.emplace<StreamToDataSet<float>>({{"filter", std::string("start")}, {"n_pre", gr::Size_t(3)}, {"n_post", gr::Size_t(4)}});
        auto& sink                       = fixture.emplace<gr::testing::DataSetCollector<float>>();
        expect(fixture.connect<"out", "in">(source, extract).has_value());
        expect(fixture.connect<"out", "in">(extract, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(sink._sets[0].signal_values, std::vector<float>{7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f}));
    };

    "a range cut at the stream's start holds only real samples and is flagged incomplete"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TaggedRamp>();
        source._tags                     = {{1UZ, named("start")}};
        source._nSamples                 = 20UZ;
        auto& extract                    = fixture.emplace<StreamToDataSet<float>>({{"filter", std::string("start")}, {"n_pre", gr::Size_t(3)}, {"n_post", gr::Size_t(4)}});
        auto& sink                       = fixture.emplace<gr::testing::DataSetCollector<float>>();
        expect(fixture.connect<"out", "in">(source, extract).has_value());
        expect(fixture.connect<"out", "in">(extract, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(sink._sets[0].signal_values, std::vector<float>{0.f, 1.f, 2.f, 3.f, 4.f})) << "no padding";
        const auto meta  = gr::property_map_view{sink._sets[0].meta_information[0]}.get_if<gr::property_map>(gr::tag::kTriggerMetaInfo.key());
        const auto flags = meta ? gr::property_map_view{*meta}.get_if<gr::property_map>(gr::tag::kAcquisitionFlags.key()) : nullptr;
        expect(flags != nullptr && gr::property_map_view{*flags}.contains(std::string_view{"incomplete"}));
    };

    "manual rearming re-opens a stopped single mode, auto does not"_test = [] {
        const std::vector<std::pair<std::size_t, gr::property_map>> tags{
            {0UZ, named("arm")},
            {2UZ, named("start")},
            {6UZ, named("arm")},
            {10UZ, named("start")},
        };
        gr::property_map manual{{"mode", std::string("single")}, {"arm_filter", std::string("arm")}, {"rearm", std::string("manual")}};
        const auto       again = capture(tags, 20UZ, manual);
        expect(eq(again.captured, 2U)) << "the second arming allows a second single range";
        const auto stays = capture(tags, 20UZ, {{"mode", std::string("single")}, {"arm_filter", std::string("arm")}});
        expect(eq(stays.captured, 1U));
    };

    "the stage walks priming, armed, awaiting samples and stopped"_test = [] {
        ManualRig rig({{"filter", std::string("start")}, {"n_pre", gr::Size_t(2)}, {"n_post", gr::Size_t(3)}, {"mode", std::string("single")}});
        auto& [extract, source, events, sink] = rig;
        std::vector<std::string> stages;
        const auto               step = [&](bool withStart) {
            if (withStart) {
                source.publishTag(gr::property_map_view{named("start")}, 0UZ);
            }
            auto samples = source.streamWriter().reserve<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            samples[0]   = 1.f;
            samples.publish(1UZ);
            std::ignore           = extract.work(1UZ);
            const std::string now = extract.stage.value;
            if (stages.empty()) {
                stages.push_back(now);
            } else if (stages.back().compare(now) != 0) {
                stages.push_back(now);
            }
        };
        step(false);
        step(false);
        step(false);
        step(true);
        step(false);
        step(false);
        step(false);
        step(false);
        expect(stages == std::vector<std::string>{"priming", "armed", "awaiting_samples", "stopped"}) << std::format("{}", stages.size());
    };

    "a dated event opens the range at the sample its trigger_time falls on, however the stream is cut"_test = [] {
        const auto firstSampleWith = [](std::size_t maxChunk) {
            gr::testing::GraphFixture fixture;
            auto&                     script = fixture.emplace<gr::testing::EventScript>();
            script._events                   = {gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("start")}, {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1'000'000'000U + 10'000'000U}}}};
            auto& source                     = fixture.emplace<TaggedRamp>();
            source._nSamples                 = 40UZ;
            source._tags                     = {{0UZ, gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("anchor")}, {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1'000'000'000U}}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}, {std::string("sample_rate"), 1000.f}}}};
            auto& extract                    = fixture.emplace<StreamToDataSet<float>>({{"filter", std::string("start")}, {"n_post", gr::Size_t(2)}});
            if (maxChunk > 0UZ) {
                extract.in.max_samples = maxChunk;
            }
            auto& sink = fixture.emplace<gr::testing::DataSetCollector<float>>();
            expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, extract, gr::PortDefinition{std::string("evtIn")}).has_value());
            expect(fixture.connect<"out", "in">(source, extract).has_value());
            expect(fixture.connect<"out", "in">(extract, sink).has_value());
            expect(fixture.run().has_value());
            return sink._sets.empty() ? -1.f : sink._sets.front().signal_values.front();
        };
        expect(eq(firstSampleWith(0UZ), 10.f)) << "10 ms after the anchor at 1 kHz is sample 10";
        expect(eq(firstSampleWith(3UZ), 10.f)) << "and in chunks of three";
        expect(eq(firstSampleWith(7UZ), 10.f)) << "or seven";
    };

    "an undated event acts at the start of the call and is counted"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<TaggedRamp>();
        source._nSamples                 = 40UZ;
        auto& script                     = fixture.emplace<gr::testing::EventScript>();
        script._events                   = {named("start")};
        auto& extract                    = fixture.emplace<StreamToDataSet<float>>({{"filter", std::string("start")}, {"n_post", gr::Size_t(2)}});
        auto& sink                       = fixture.emplace<gr::testing::DataSetCollector<float>>();
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, extract, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.connect<"out", "in">(source, extract).has_value());
        expect(fixture.connect<"out", "in">(extract, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(extract._undatedEvents, 1U));
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

constexpr std::uint64_t kAnchorNs = 1'000'000'000U;

/** sample i of the ramp carries the value i; the anchor dates sample 0 at kAnchorNs, at 1 kHz */
void publishRamp(gr::PortOut<float>& port, std::size_t& next, std::size_t n) {
    if (next == 0UZ) {
        const gr::property_map anchor{{gr::testing::kTriggerKey, std::string("anchor")}, {std::string(gr::tag::kTriggerTime.key()), kAnchorNs}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}, {std::string("sample_rate"), 1000.f}};
        port.publishTag(gr::property_map_view{anchor}, 0UZ);
    }
    auto span = port.streamWriter().reserve<gr::SpanReleasePolicy::ProcessAll>(n);
    std::iota(span.begin(), span.end(), static_cast<float>(next));
    span.publish(n);
    next += n;
}

void emitDated(gr::EventPortOut& events, std::string name, std::size_t sample) {
    const gr::property_map event{{gr::testing::kTriggerKey, std::move(name)}, {std::string(gr::tag::kTriggerTime.key()), kAnchorNs + sample * 1'000'000U}};
    auto                   span = events.reserve<gr::SpanReleasePolicy::ProcessNone>(1UZ);
    boost::ut::expect(gr::emit(span, 0UZ, gr::property_map_view{event}).has_value()) << boost::ut::fatal;
    span.publish(1UZ);
}

/** the block takes `takenBefore` samples, then a `start` dated at `startAt` arrives, then 600 more samples */
void feedLateStart(auto& block, gr::PortOut<float>& source, gr::EventPortOut& events, std::size_t takenBefore, std::size_t startAt) {
    std::size_t next = 0UZ;
    publishRamp(source, next, takenBefore);
    std::ignore = block.work(takenBefore);
    emitDated(events, "start", startAt);
    publishRamp(source, next, 600UZ);
    for (int call = 0; call < 3; ++call) {
        std::ignore = block.work(600UZ);
    }
}

enum class Arrival { early, sameSpan, late };

struct EqualCut {
    std::size_t                                       startSample  = 0UZ;
    double                                            eventSamples = 0.; /// the trigger's moment relative to startSample, in (-1, 0]
    Arrival                                           arrival      = Arrival::early;
    std::size_t                                       span         = 1024UZ;
    std::vector<std::pair<std::size_t, std::int64_t>> timingJitterNs; /// timing tags: sample, and how far its trigger_time is off the sample count
};

/** the first sample and size of the range a dated `start` opens; the event is dated from the stream's own timing tags, as an upstream trigger at `startSample` would date it */
[[nodiscard]] std::pair<float, std::size_t> datedCut(const EqualCut& cut) {
    constexpr std::size_t kSamples = 4096UZ;
    ManualRig             rig({{"filter", std::string("start")}, {"n_pre", gr::Size_t(100)}, {"n_post", gr::Size_t(50)}});
    const auto            timingTag = [&](std::size_t index) {
        const auto jitter = std::ranges::find(cut.timingJitterNs, index, &std::pair<std::size_t, std::int64_t>::first);
        if (jitter == cut.timingJitterNs.end()) {
            return std::optional<gr::property_map>{};
        }
        const auto ns = static_cast<std::uint64_t>(static_cast<std::int64_t>(kAnchorNs + index * 1'000'000U) + jitter->second);
        return std::optional{gr::property_map{{std::string(gr::tag::kTriggerTime.key()), ns}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}, {std::string("sample_rate"), 1000.f}}};
    };
    gr::blocks::trigger::TimeBase upstream;
    upstream.setRate(1000.);
    for (std::size_t i = 0UZ; i <= cut.startSample; ++i) {
        if (const auto tag = timingTag(i)) {
            std::ignore = upstream.adopt(gr::property_map_view{*tag}, i);
        }
    }
    const gr::property_map event = gr::blocks::trigger::datedTrigger(upstream, cut.startSample, cut.eventSamples, 0.f, "start", "");
    const auto             emit  = [&] {
        auto span = rig.events.reserve<gr::SpanReleasePolicy::ProcessNone>(1UZ);
        boost::ut::expect(gr::emit(span, 0UZ, gr::property_map_view{event}).has_value()) << boost::ut::fatal;
        span.publish(1UZ);
    };
    bool emitted = false;
    if (cut.arrival == Arrival::early) {
        emit();
        emitted = true;
    }
    for (std::size_t from = 0UZ; from < kSamples; from += cut.span) {
        const std::size_t n = std::min(cut.span, kSamples - from);
        if (cut.arrival == Arrival::sameSpan && !emitted && cut.startSample < from + n) {
            emit();
            emitted = true;
        }
        {
            for (std::size_t i = from; i < from + n; ++i) {
                if (const auto tag = timingTag(i)) {
                    rig.source.publishTag(gr::property_map_view{*tag}, i - from);
                }
            }
            auto samples = rig.source.streamWriter().reserve<gr::SpanReleasePolicy::ProcessAll>(n);
            std::iota(samples.begin(), samples.end(), static_cast<float>(from));
            samples.publish(n);
        }
        for (int call = 0; call < 64 && rig.extract.in.streamReader().available() > 0UZ; ++call) {
            std::ignore = rig.extract.work(n);
        }
        if (cut.arrival == Arrival::late && !emitted && from + n > cut.startSample) {
            boost::ut::expect(boost::ut::le(from + n - cut.startSample, StreamToDataSet<float>::MIN_BUFFER_SIZE)) << boost::ut::fatal << "late within the held samples";
            emit();
            emitted = true;
        }
    }
    auto sets = rig.sink.streamReader().get<gr::SpanReleasePolicy::ProcessAll>(rig.sink.streamReader().available());
    if (sets.empty() || sets[0].signal_values.empty()) {
        return {-1.f, 0UZ};
    }
    return {sets[0].signal_values.front(), sets[0].signal_values.size()};
}

const auto lateEvents = [] {
    using namespace boost::ut;

    "a dated start opens on the sample an in-stream trigger of the same moment sits on, early, in its span or late"_test = [] {
        const std::vector<std::pair<std::size_t, std::int64_t>> exact{{0UZ, 0}, {1000UZ, 0}, {2000UZ, 0}, {3000UZ, 0}};
        const std::vector<std::pair<std::size_t, std::int64_t>> wallClock{{0UZ, 0}, {1000UZ, -6'000'000}, {2000UZ, 12'000'000}, {3000UZ, -5'000'000}};
        for (const auto& [timing, timingName] : {std::pair{exact, "exact tags"}, std::pair{wallClock, "tags off by -6, +12, -5 ms"}}) {
            for (const std::size_t span : {480UZ, 1024UZ}) {
                for (const std::size_t startSample : {1920UZ, 2020UZ, 2944UZ}) {
                    for (const double eventSamples : {0., -0.3, -0.7}) {
                        for (const Arrival arrival : {Arrival::early, Arrival::sameSpan, Arrival::late}) {
                            const auto [first, size] = datedCut({.startSample = startSample, .eventSamples = eventSamples, .arrival = arrival, .span = span, .timingJitterNs = timing});
                            expect(eq(first, static_cast<float>(startSample - 100UZ)) && eq(size, 150UZ)) << std::format("{}, spans of {}, start on {} ({:+.1f}), event {}: the range [start - n_pre, start + n_post)", timingName, span, startSample, eventSamples, gr::meta::enumName(arrival).value_or(""));
                        }
                    }
                }
            }
        }
    };

    "a dated start arriving after its sample opens the range on that sample, n_pre before it"_test = [] {
        for (const std::size_t lateBy : {0UZ, 50UZ, 150UZ}) {
            ManualRig rig({{"filter", std::string("start")}, {"n_pre", gr::Size_t(100)}, {"n_post", gr::Size_t(50)}});
            feedLateStart(rig.extract, rig.source, rig.events, 400UZ, 400UZ - lateBy);

            auto sets = rig.sink.streamReader().get<gr::SpanReleasePolicy::ProcessAll>(rig.sink.streamReader().available());
            expect(eq(sets.size(), 1UZ)) << fatal << "late by" << lateBy;
            const auto expected = std::views::iota(0UZ, 150UZ) | std::views::transform([lateBy](std::size_t i) { return static_cast<float>(300UZ - lateBy + i); }) | std::ranges::to<std::vector<float>>();
            expect(std::ranges::equal(sets[0].signal_values, expected)) << "the samples [start - n_pre, start + n_post), late by" << lateBy;
            expect(eq(sets[0].axis_values[0].front(), -0.1f)) << "the first sample 100 samples before the start at 1 kHz, late by" << lateBy;
            expect(eq(rig.extract._lateEvents, 0U));
        }
    };

    "an event source that finishes and disconnects evtIn does not stop the block keeping samples for late events"_test = [] {
        ManualRig rig({{"filter", std::string("start")}, {"n_post", gr::Size_t(50)}});
        expect(rig.extract._eventsConnected) << "evtIn connected at start";
        std::ignore = rig.extract.evtIn.disconnect(); // what a finished source leaves behind
        expect(!rig.extract.evtIn.isConnected() && rig.extract._eventsConnected) << "still counted as connected: decided at start, so held samples stay for its late events";
    };

    "without n_pre a late start still opens on its own sample"_test = [] {
        ManualRig rig({{"filter", std::string("start")}, {"n_post", gr::Size_t(50)}});
        feedLateStart(rig.extract, rig.source, rig.events, 400UZ, 250UZ);

        auto sets = rig.sink.streamReader().get<gr::SpanReleasePolicy::ProcessAll>(rig.sink.streamReader().available());
        expect(eq(sets.size(), 1UZ)) << fatal;
        expect(eq(sets[0].signal_values.size(), 50UZ));
        expect(eq(sets[0].signal_values.front(), 250.f));
        expect(eq(sets[0].signal_values.back(), 299.f)) << "the whole range lies in the held samples";
    };

    "a start older than the held samples opens at the call start and is counted"_test = [] {
        ManualRig rig({{"filter", std::string("start")}, {"n_pre", gr::Size_t(10)}, {"n_post", gr::Size_t(5)}});
        feedLateStart(rig.extract, rig.source, rig.events, 2000UZ, 100UZ);

        auto sets = rig.sink.streamReader().get<gr::SpanReleasePolicy::ProcessAll>(rig.sink.streamReader().available());
        expect(eq(sets.size(), 1UZ)) << fatal;
        expect(eq(sets[0].signal_values.front(), 1990.f)) << "n_pre before the first sample of the call it acted in";
        expect(eq(rig.extract._lateEvents, 1U));
    };

    "a late stop acts at once and is counted"_test = [] {
        ManualRig   rig({{"filter", std::string("[start, stop]")}});
        std::size_t next = 0UZ;
        publishRamp(rig.source, next, 100UZ);
        emitDated(rig.events, "start", 20UZ);
        std::ignore = rig.extract.work(100UZ);
        publishRamp(rig.source, next, 100UZ);
        std::ignore = rig.extract.work(100UZ);
        std::ignore = rig.extract.work(100UZ); // the block has now taken all 200 samples
        emitDated(rig.events, "stop", 150UZ);
        publishRamp(rig.source, next, 100UZ);
        std::ignore = rig.extract.work(100UZ);

        auto sets = rig.sink.streamReader().get<gr::SpanReleasePolicy::ProcessAll>(rig.sink.streamReader().available());
        expect(eq(sets.size(), 1UZ)) << fatal;
        expect(eq(sets[0].signal_values.size(), 180UZ)) << "the range [20, 200): the stop acted where it was found, not on sample 150";
        expect(eq(rig.extract._lateEvents, 1U)) << "the stop's sample had passed";
    };

    "StreamFilter also opens a late start on its own sample"_test = [] {
        StreamFilter<float> filter({{"filter", std::string("start")}, {"n_pre", gr::Size_t(100)}, {"n_post", gr::Size_t(50)}});
        gr::PortOut<float>  source;
        gr::EventPortOut    events;
        gr::PortIn<float>   sink;
        expect(source.connect(filter.in).has_value()) << fatal;
        expect(events.connect(filter.evtIn).has_value()) << fatal;
        expect(filter.out.connect(sink).has_value()) << fatal;
        filter.init(filter.progress);
        expect(filter.changeStateTo(gr::lifecycle::State::RUNNING).has_value()) << fatal;
        feedLateStart(filter, source, events, 400UZ, 250UZ);

        auto samples = sink.streamReader().get<gr::SpanReleasePolicy::ProcessAll>(sink.streamReader().available());
        expect(eq(samples.size(), 150UZ)) << fatal;
        expect(eq(samples.front(), 150.f)) << "n_pre before the start's own sample";
        expect(eq(samples.back(), 299.f));
        expect(eq(filter._lateEvents, 0U));
    };
};

} // namespace

int main() {
    using namespace boost::ut;
    "StreamToDataSet arming, holdoff and segment limit"_test = arming;
    "StreamToDataSet late dated events"_test                 = lateEvents;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
