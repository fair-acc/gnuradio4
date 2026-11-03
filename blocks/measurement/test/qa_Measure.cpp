#include <boost/ut.hpp>

#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gnuradio-4.0/measurement/EventStatistics.hpp>
#include <gnuradio-4.0/measurement/Measure.hpp>
#include <gnuradio-4.0/measurement/MeasurementCondition.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/TimeInterval.hpp>
#include <gnuradio-4.0/trigger/ValueTrigger.hpp>

#include "TriggerTest.hpp"

/** nanoseconds per millisecond: one sample at 1 kHz */
constexpr std::uint64_t kMs = 1'000'000U;

namespace qa_measure {
using gr::trigger_test::kStreamStart;
using namespace gr::blocks::measurement;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;

struct Result {
    std::optional<double> value;
    std::string           unit;
    std::string           reason;
    std::string           opener;           /// the trigger_name of the event that opened the gate
    std::uint64_t         start       = 0U; /// trigger_meta_info.trigger_time: the gate's first sample
    std::uint64_t         stop        = 0U; /// trigger_time: the gate's last sample, where it was decided
    float                 offset      = -1.f;
    float                 startOffset = -1.f;  /// trigger_meta_info.trigger_offset
    bool                  oldKeys     = false; /// trigger_stop_time, valid or a source, at the root or in trigger_meta_info
    gr::Size_t            channel     = 0U;
    gr::Size_t            samples     = 0U;
};

[[nodiscard]] std::vector<Result> resultsOf(const EventTap& tap) {
    std::vector<Result> found;
    for (const gr::property_map& event : tap._events) {
        const gr::property_map_view view{event};
        const auto                  meta = view.get_if<gr::property_map>(gr::tag::kTriggerMetaInfo.key());
        if (!meta) {
            continue;
        }
        const gr::property_map_view details{*meta};
        Result                      result;
        if (const auto value = details.get_if<double>(std::string_view{"value"})) {
            result.value = *value;
        }
        if (const auto unit = details.get_if<std::string_view>(std::string_view{"unit"})) {
            result.unit = std::string(*unit);
        }
        if (const auto reason = details.get_if<std::string_view>(std::string_view{"reason"})) {
            result.reason = std::string(*reason);
        }
        if (const auto opener = details.get_if<std::string_view>(gr::tag::kTriggerName.key())) {
            result.opener = std::string(*opener);
        }
        result.start       = details.get_if<std::uint64_t>(gr::tag::kTriggerTime.key()) ? *details.get_if<std::uint64_t>(gr::tag::kTriggerTime.key()) : 0U;
        result.stop        = view.get_if<std::uint64_t>(gr::tag::kTriggerTime.key()) ? *view.get_if<std::uint64_t>(gr::tag::kTriggerTime.key()) : 0U;
        result.startOffset = details.get_if<float>(gr::tag::kTriggerOffset.key()) ? *details.get_if<float>(gr::tag::kTriggerOffset.key()) : -1.f;
        result.offset      = view.get_if<float>(gr::tag::kTriggerOffset.key()) ? *view.get_if<float>(gr::tag::kTriggerOffset.key()) : -1.f;
        result.oldKeys     = view.contains(gr::tag::kTriggerStopTime.key()) || view.contains(std::string_view{"source"}) || details.contains(std::string_view{"source"}) || details.contains(std::string_view{"valid"});
        result.channel     = details.get_if<gr::Size_t>(std::string_view{"channel"}) ? *details.get_if<gr::Size_t>(std::string_view{"channel"}) : 0U;
        result.samples     = details.get_if<gr::Size_t>(std::string_view{"n_samples"}) ? *details.get_if<gr::Size_t>(std::string_view{"n_samples"}) : 0U;
        found.push_back(result);
    }
    return found;
}

/** a 1 kHz stream dated 1 s at its first sample, in `unit`; `events` gate it, `settings` choose the gates */
template<Quantity quantity>
[[nodiscard]] std::vector<Result> measure(std::vector<float> samples, std::vector<gr::property_map> events, gr::property_map settings, std::vector<float>* passed = nullptr, std::vector<std::pair<std::size_t, gr::property_map>> extraTags = {}, std::size_t chunk = 0UZ, std::uint32_t seed = 0U, std::size_t* nGaps = nullptr) {
    gr::testing::GraphFixture fixture;
    settings[std::string("sample_rate")] = 1000.f;
    const gr::property_map anchor{
        {std::string(gr::tag::kTriggerTime.key()), kStreamStart},
        {std::string(gr::tag::kTriggerOffset.key()), 0.f},
        {std::string(gr::tag::kSignalUnit.key()), std::string("V")},
    };
    const auto play = [&](auto& source) {
        auto& script           = fixture.emplace<EventScript>();
        script._events         = std::move(events);
        script._stopWhenDone   = false;
        script._maxPerWorkCall = seed > 0U ? 1UZ + seed % 3UZ : std::numeric_limits<std::size_t>::max();
        auto& block            = fixture.emplace<Measure<float, quantity>>(std::move(settings));
        auto& sink             = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        auto& tap              = fixture.emplace<EventTap>();
        if (chunk > 0UZ) {
            for (auto& port : block.in) {
                port.max_samples = chunk;
            }
        }
        boost::ut::expect(fixture.graph.connect(source, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in#0")}).has_value());
        boost::ut::expect(fixture.graph.connect(block, gr::PortDefinition{std::string("out#0")}, sink, gr::PortDefinition{std::string("in")}).has_value());
        boost::ut::expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, block, gr::PortDefinition{std::string("evtIn")}).has_value());
        boost::ut::expect(fixture.graph.connect(block, gr::PortDefinition{std::string("evtOut")}, tap, gr::PortDefinition{std::string("evtIn")}).has_value());
        boost::ut::expect(fixture.run().has_value());
        if (passed != nullptr) {
            passed->assign(sink._samples.begin(), sink._samples.end());
        }
        if (nGaps != nullptr) {
            *nGaps = block.n_gaps.value;
        }
        return resultsOf(tap);
    };
    if (seed > 0U) {
        auto& source    = fixture.emplace<gr::trigger_test::PartitionSource<float>>();
        source._samples = std::move(samples);
        source._tags    = std::move(extraTags);
        source._tags.insert(source._tags.begin(), {0UZ, anchor});
        source._seed = seed;
        return play(source);
    }
    auto& source = fixture.emplace<gr::testing::TagSource<float>>({
        {"values", samples},
        {"n_samples_max", static_cast<gr::Size_t>(samples.size())},
        {"mark_tag", false},
    });
    source._tags.push_back({0UZ, anchor});
    for (auto& [at, tag] : extraTags) {
        source._tags.push_back({at, std::move(tag)});
    }
    return play(source);
}

const boost::ut::suite<"Measure (IEEE 181-2011)"> _measure = [] {
    using namespace boost::ut;

    "without an open filter the stream is cut into back-to-back gates of length"_test = [] {
        const auto found = measure<Quantity::mean>({1.f, 1.f, 1.f, 1.f, 3.f, 3.f, 3.f, 3.f}, {}, {{"length", 4U}});
        expect(eq(found.size(), 2UZ)) << fatal;
        expect(found[0].value == std::optional{1.} && found[1].value == std::optional{3.});
        expect(eq(found[0].unit, std::string("V"))) << "an amplitude keeps the stream's unit";
        expect(eq(found[0].samples, 4U));
        expect(eq(found[0].start, kStreamStart) && eq(found[0].stop, kStreamStart + 3U * kMs)) << "the gate covers samples 0 to 3: meta start, decided at the last";
        expect(eq(found[1].start, kStreamStart + 4U * kMs) && eq(found[1].stop, kStreamStart + 7U * kMs));
        expect(found[0].offset == 0.f && found[1].offset == 0.f) << "the decision sample is on the grid";
        expect(found[0].startOffset == 0.f && found[1].startOffset == 0.f) << "the start tuple carries trigger_offset too";
        expect(!found[0].oldKeys && !found[1].oldKeys) << "no trigger_stop_time, valid or source any more";
    };

    "a reset by the scheduler clears the gates, the pending results and the counters"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<gr::testing::TagSource<float>>({{"values", std::vector<float>{1.f, 1.f, 1.f, 1.f, 3.f, 3.f, 3.f, 3.f}}, {"n_samples_max", 8U}, {"mark_tag", false}});
        auto&                     block  = fixture.emplace<Measure<float, Quantity::mean>>({{"length", 4U}, {"sample_rate", 1000.f}});
        auto&                     sink   = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        auto&                     script = fixture.emplace<EventScript>();
        auto&                     tap    = fixture.emplace<EventTap>();
        script._stopWhenDone             = false;
        expect(fixture.graph.connect(source, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(block, gr::PortDefinition{std::string("out#0")}, sink, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, block, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(block, gr::PortDefinition{std::string("evtOut")}, tap, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.run().has_value());
        expect(eq(block.n_measured.value, 2U)) << fatal;

        expect(block.changeStateTo(gr::lifecycle::State::INITIALISED).has_value()) << fatal;
        expect(eq(block.n_measured.value, 0U)) << "the scheduler's INITIALISED transition calls reset()";
    };

    "an event opens a gate n_pre before and n_post after it, and a width comes out in seconds"_test = [] {
        // the gate [1, 7) holds 0 0 1 1 1 0: a positive pulse of 3 samples at the 50 % level, 3 ms at 1 kHz
        const auto found = measure<Quantity::positiveWidth>({0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f}, {eventNamed("pulse", kStreamStart + 3U * kMs)}, {{"open_filter", std::string("pulse")}, {"n_pre", 2U}, {"n_post", 4U}});
        expect(eq(found.size(), 1UZ)) << fatal;
        expect(found[0].value.has_value() && approx(*found[0].value, 3e-3, 1e-12));
        expect(eq(found[0].unit, std::string("s")));
        expect(eq(found[0].start, kStreamStart + kMs)) << "n_pre = 2 before the event at sample 3";
        expect(eq(found[0].stop, kStreamStart + 6U * kMs)) << "n_post = 4 is exclusive: the gate [1, 7) ends at sample 6";
        expect(eq(found[0].opener, std::string("pulse"))) << "the gate start is named for the event that opened it";
    };

    "a close event ends the gate an open event began"_test = [] {
        const auto found = measure<Quantity::maximum>({0.f, 0.f, 2.f, 5.f, 2.f, 0.f, 9.f, 0.f}, {eventNamed("open", kStreamStart + 2U * kMs), eventNamed("close", kStreamStart + 5U * kMs)}, {{"open_filter", std::string("open")}, {"close_filter", std::string("close")}});
        expect(eq(found.size(), 1UZ)) << fatal;
        expect(found[0].value == std::optional{5.}) << "samples 2 to 4; the 9 after the close is not in it";
        expect(eq(found[0].start, kStreamStart + 2U * kMs)) << "opened by the event at sample 2";
        expect(eq(found[0].opener, std::string("open")));
    };

    "a gate the stream's start cut short is censored, not measured"_test = [] {
        const auto found = measure<Quantity::maximum>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {eventNamed("e", kStreamStart + 1U * kMs)}, {{"open_filter", std::string("e")}, {"n_pre", 3U}, {"n_post", 1U}});
        expect(eq(found.size(), 1UZ)) << fatal;
        expect(!found[0].value.has_value());
        expect(eq(found[0].reason, std::string("censored")));
    };

    "a gate no close event ends is cut at max_length and reported truncated"_test = [] {
        const auto found = measure<Quantity::mean>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, {eventNamed("open", kStreamStart + 1U * kMs)}, {{"open_filter", std::string("open")}, {"close_filter", std::string("close")}, {"max_length", 4U}});
        expect(eq(found.size(), 1UZ)) << fatal;
        expect(eq(found[0].reason, std::string("truncated")));
        expect(eq(found[0].samples, 4U));
    };

    "a gate holding a non-finite sample is reported invalid, with its reason and without a value"_test = [] {
        const auto found = measure<Quantity::rms>({1.f, std::numeric_limits<float>::quiet_NaN(), 1.f, 1.f}, {}, {{"length", 4U}});
        expect(eq(found.size(), 1UZ)) << fatal;
        expect(!found[0].value.has_value());
        expect(eq(found[0].reason, std::string("nonFinite")));
    };

    "the samples pass through unchanged"_test = [] {
        std::vector<float> passed;
        const std::vector  samples{1.f, 2.f, 3.f, 4.f, 5.f};
        std::ignore = measure<Quantity::area>(samples, {}, {{"length", 2U}}, &passed);
        expect(passed == samples);
    };

    "each channel is measured over the same gate, and its result is tagged on its own output"_test = [] {
        gr::testing::GraphFixture fixture;
        const std::vector<float>  low{1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
        const std::vector<float>  high{4.f, 4.f, 4.f, 4.f, 4.f, 4.f};
        auto&                     sourceA = fixture.emplace<gr::testing::TagSource<float>>({{"values", low}, {"n_samples_max", gr::Size_t{6U}}, {"mark_tag", false}});
        auto&                     sourceB = fixture.emplace<gr::testing::TagSource<float>>({{"values", high}, {"n_samples_max", gr::Size_t{6U}}, {"mark_tag", false}});
        sourceA._tags.push_back({0UZ, gr::property_map{{std::string(gr::tag::kTriggerTime.key()), kStreamStart}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}});
        auto& block = fixture.emplace<Measure<float, Quantity::rms>>({{"n_inputs", 2U}, {"length", 3U}, {"sample_rate", 1000.f}});
        auto& sinkA = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        auto& sinkB = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        auto& tap   = fixture.emplace<EventTap>();
        expect(fixture.graph.connect(sourceA, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(sourceB, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in#1")}).has_value());
        expect(fixture.graph.connect(block, gr::PortDefinition{std::string("out#0")}, sinkA, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.graph.connect(block, gr::PortDefinition{std::string("out#1")}, sinkB, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.graph.connect(block, gr::PortDefinition{std::string("evtOut")}, tap, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.run().has_value());

        const auto found = resultsOf(tap);
        expect(eq(found.size(), 4UZ)) << "two gates of three samples, two channels";
        expect(std::ranges::count_if(found, [](const Result& r) { return r.channel == 1U && r.value == std::optional{4.}; }) == 2) << "channel 1 is 4 throughout";
        const auto tagged = [](const auto& sink) { return std::ranges::count_if(sink._tags, [](const auto& entry) { return gr::property_map_view{entry.map}.contains(gr::tag::kTriggerMetaInfo.key()); }); };
        expect(ge(tagged(sinkA) + tagged(sinkB), 2)) << "the results also travel in-band, on each channel's own output";
    };

    "the sample and the population standard deviation are two explicit quantities"_test = [] {
        const std::vector<float> set{2.f, 4.f, 4.f, 4.f, 5.f, 5.f, 7.f, 9.f}; // mean 5, sum of squared deviations 32
        const auto               sample     = measure<Quantity::sampleStdDev>(set, {}, {{"length", 8U}});
        const auto               population = measure<Quantity::populationStdDev>(set, {}, {{"length", 8U}});
        expect(eq(sample.size(), 1UZ) and eq(population.size(), 1UZ)) << fatal;
        expect(sample[0].value.has_value() && approx(*sample[0].value, std::sqrt(32. / 7.), 1e-6));
        expect(population[0].value.has_value() && approx(*population[0].value, 2., 1e-6));
        expect(eq(sample[0].unit, std::string("V")));
    };

    "the period and frequency of a square wave of eight samples per cycle at 1 kHz are 8 ms and 125 Hz"_test = [] {
        const std::vector<float> square{0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f};
        const auto               period    = measure<Quantity::period>(square, {}, {{"length", 24U}});
        const auto               frequency = measure<Quantity::frequency>(square, {}, {{"length", 24U}});
        expect(eq(period.size(), 1UZ) and eq(frequency.size(), 1UZ)) << fatal;
        expect(period[0].value.has_value() && approx(*period[0].value, 8e-3, 1e-9));
        expect(eq(period[0].unit, std::string("s")));
        expect(frequency[0].value.has_value() && approx(*frequency[0].value, 125., 1e-6));
        expect(eq(frequency[0].unit, std::string("Hz")));
    };

    "the delay and phase of channel 1 against channel 0 are taken between their rising mid-level crossings"_test = [] {
        const std::vector<float> leader{0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f};
        std::vector<float>       lagging(2UZ, 0.f); // two samples later: a quarter of the eight-sample cycle
        lagging.insert(lagging.end(), leader.begin(), leader.end() - 2);
        const auto bothChannels = [&]<Quantity quantity>() {
            gr::testing::GraphFixture fixture;
            auto&                     sourceA = fixture.emplace<gr::testing::TagSource<float>>({
                {"values", leader},
                {"n_samples_max", static_cast<gr::Size_t>(leader.size())},
                {"mark_tag", false},
            });
            auto&                     sourceB = fixture.emplace<gr::testing::TagSource<float>>({
                {"values", lagging},
                {"n_samples_max", static_cast<gr::Size_t>(lagging.size())},
                {"mark_tag", false},
            });
            sourceA._tags.push_back({0UZ, gr::property_map{{std::string(gr::tag::kTriggerTime.key()), kStreamStart}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}});
            auto& block = fixture.emplace<Measure<float, quantity>>({{"length", 24U}, {"sample_rate", 1000.f}});
            auto& sinkA = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
            auto& sinkB = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
            auto& tap   = fixture.emplace<EventTap>();
            expect(fixture.graph.connect(sourceA, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in#0")}).has_value());
            expect(fixture.graph.connect(sourceB, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in#1")}).has_value());
            expect(fixture.graph.connect(block, gr::PortDefinition{std::string("out#0")}, sinkA, gr::PortDefinition{std::string("in")}).has_value());
            expect(fixture.graph.connect(block, gr::PortDefinition{std::string("out#1")}, sinkB, gr::PortDefinition{std::string("in")}).has_value());
            expect(fixture.graph.connect(block, gr::PortDefinition{std::string("evtOut")}, tap, gr::PortDefinition{std::string("evtIn")}).has_value());
            expect(fixture.run().has_value());
            return resultsOf(tap);
        };
        const auto delay = bothChannels.template operator()<Quantity::delay>();
        const auto phase = bothChannels.template operator()<Quantity::phase>();
        expect(eq(delay.size(), 1UZ) and eq(phase.size(), 1UZ)) << fatal << "one result per gate for the pair";
        expect(delay[0].value.has_value() && approx(*delay[0].value, 2e-3, 1e-9)) << "two samples at 1 kHz";
        expect(eq(delay[0].unit, std::string("s")));
        expect(phase[0].value.has_value() && approx(*phase[0].value, 90., 1e-6)) << "a quarter cycle";
        expect(eq(phase[0].unit, std::string("deg")));
    };

    "a ramp has no top: the gate reports why instead of a mid-level"_test = [] {
        const auto found = measure<Quantity::top>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}, {}, {{"length", 8U}});
        expect(eq(found.size(), 1UZ)) << fatal;
        expect(!found[0].value.has_value());
        expect(eq(found[0].reason, std::string("noTwoStates")));
    };

    "a gap inside a gate censors that gate only and nothing is measured across it"_test = [] {
        const gr::property_map gap{{std::string(gr::tag::kNDroppedSamples.key()), gr::Size_t{100U}}};
        const auto             found = measure<Quantity::mean>(
            {
                1.f,
                1.f,
                1.f,
                1.f,
                2.f,
                2.f,
                2.f,
                2.f,
                3.f,
                3.f,
                3.f,
                3.f,
            },
            {}, {{"length", 4U}}, nullptr, {{5UZ, gap}});
        expect(eq(found.size(), 3UZ)) << fatal;
        expect(found[0].value == std::optional{1.} && found[2].value == std::optional{3.}) << "the gates either side are whole";
        expect(!found[1].value.has_value());
        expect(eq(found[1].reason, std::string("censored")));
    };

    "the results do not depend on how the scheduler cuts the stream"_test = [] {
        std::vector<float> wave;
        for (std::size_t i = 0UZ; i < 48UZ; ++i) {
            wave.push_back(static_cast<float>((i * 7UZ) % 11UZ));
        }
        const std::vector<gr::property_map> events{
            eventNamed("open", kStreamStart + 6U * kMs),
            eventNamed("open", kStreamStart + 20U * kMs),
            eventNamed("open", kStreamStart + 33U * kMs),
        };
        const gr::property_map gated{{"open_filter", std::string("open")}, {"n_pre", 2U}, {"n_post", 5U}};
        const auto             whole = measure<Quantity::acRms>(wave, events, gated);
        expect(eq(whole.size(), 3UZ)) << fatal;
        for (const std::size_t chunk : {1UZ, 3UZ, 5UZ, 7UZ, 16UZ}) {
            const auto cut = measure<Quantity::acRms>(wave, events, gated, nullptr, {}, chunk);
            expect(eq(cut.size(), whole.size())) << fatal << "chunk " << chunk;
            for (std::size_t i = 0UZ; i < whole.size(); ++i) {
                expect(cut[i].value == whole[i].value and cut[i].start == whole[i].start and cut[i].stop == whole[i].stop) << "gate " << i << " with chunk " << chunk;
            }
        }
        const auto ungatedWhole = measure<Quantity::mean>(wave, {}, {{"length", 10U}});
        const auto ungatedCut   = measure<Quantity::mean>(wave, {}, {{"length", 10U}}, nullptr, {}, 3UZ);
        expect(eq(ungatedWhole.size(), 4UZ) and eq(ungatedCut.size(), 4UZ));
        expect(std::ranges::equal(ungatedWhole, ungatedCut, [](const Result& a, const Result& b) { return a.value == b.value and a.stop == b.stop; }));
    };

    "gates are measured to the analytic value however stream and events are cut, a gap censoring the one it falls in"_test = [] {
        std::vector<float> ramp;
        for (std::size_t i = 0UZ; i < 48UZ; ++i) {
            ramp.push_back(static_cast<float>(i));
        }
        const std::vector<gr::property_map> events{
            eventNamed("open", kStreamStart + 6U * kMs),
            eventNamed("open", kStreamStart + 20U * kMs),
            eventNamed("open", kStreamStart + 33U * kMs),
        };
        const gr::property_map gated{{"open_filter", std::string("open")}, {"n_pre", 2U}, {"n_post", 5U}};
        const gr::property_map gap{{std::string(gr::tag::kNDroppedSamples.key()), gr::Size_t{100U}}};
        // each gate runs from n_pre before the event to n_post after it, n_post exclusive, as in the gate test above: samples 4..10, 18..24, 31..37 of a ramp, whose mean is the central sample
        for (const bool gapped : {false, true}) {
            for (const auto& [chunk, seed] : std::vector<std::pair<std::size_t, std::uint32_t>>{
                     {0UZ, 0U},
                     {1UZ, 0U},
                     {2UZ, 0U},
                     {3UZ, 0U},
                     {5UZ, 0U},
                     {7UZ, 0U},
                     {0UZ, 1U},
                     {0UZ, 2U},
                     {0UZ, 3U},
                     {0UZ, 4U},
                     {4UZ, 5U},
                     {2UZ, 6U},
                 }) {
                std::size_t nGaps = 0UZ;
                const auto  found = measure<Quantity::mean>(ramp, events, gated, nullptr, gapped ? std::vector<std::pair<std::size_t, gr::property_map>>{{22UZ, gap}} : std::vector<std::pair<std::size_t, gr::property_map>>{}, chunk, seed, &nGaps);
                const auto  where = std::format("chunk {}, seed {}, gap {}", chunk, seed, gapped);
                expect(eq(found.size(), 3UZ)) << fatal << where;
                expect(found[0].value == std::optional{7.0} and found[2].value == std::optional{34.0}) << where;
                expect(eq(found[0].start, kStreamStart + 4U * kMs) and eq(found[0].stop, kStreamStart + 10U * kMs) and eq(found[2].start, kStreamStart + 31U * kMs) and eq(found[2].stop, kStreamStart + 37U * kMs)) << where;
                if (gapped) {
                    expect(!found[1].value.has_value() and eq(found[1].reason, std::string("censored")) and eq(nGaps, 1UZ)) << where;
                } else {
                    expect(found[1].value == std::optional{21.0} and eq(nGaps, 0UZ)) << where;
                }
            }
        }
    };

    "the period of a square wave by composition: a level trigger into a time interval"_test = [] {
        gr::testing::GraphFixture fixture;
        std::vector<float>        square;
        for (std::size_t i = 0UZ; i < 64UZ; ++i) {
            square.push_back((i / 4UZ) % 2UZ == 0UZ ? 0.f : 1.f); // a period of 8 samples, 8 ms at 1 kHz
        }
        auto& source = fixture.emplace<gr::testing::TagSource<float>>({{"values", square}, {"n_samples_max", static_cast<gr::Size_t>(square.size())}, {"mark_tag", false}});
        source._tags.push_back({0UZ, gr::property_map{{std::string(gr::tag::kTriggerTime.key()), kStreamStart}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}});
        auto& edges   = fixture.emplace<gr::blocks::trigger::ValueTrigger<float, gr::blocks::trigger::ValueCondition::level>>({{"threshold", 0.5f}, {"sample_rate", 1000.f}, {"trigger_name", std::string("edge")}});
        auto& period  = fixture.emplace<gr::blocks::trigger::TimeInterval<double>>({{"mode", std::string("toPrevious")}, {"measure_filter", std::string("edge")}});
        auto& sink    = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        auto& periods = fixture.emplace<gr::testing::TagSink<double, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.connect<"out", "in">(source, edges).has_value());
        expect(fixture.connect<"out", "in">(edges, sink).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(edges, period).has_value());
        expect(fixture.connect<"out", "in">(period, periods).has_value());
        expect(fixture.run().has_value());
        expect(ge(periods._samples.size(), 6UZ));
        expect(std::ranges::all_of(periods._samples, [](double seconds) { return std::abs(seconds - 8e-3) < 1e-9; })) << "every rising edge is 8 ms after the one before";
    };
};
} // namespace qa_measure

namespace qa_measurement_condition {
using namespace gr::blocks::measurement;
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;

[[nodiscard]] gr::property_map resultWith(std::optional<double> value) {
    gr::property_map details;
    if (value) {
        details[std::string("value")] = *value;
    }
    return gr::property_map{{std::string(gr::tag::kTriggerName.key()), std::string("rms")}, {std::string(gr::tag::kTriggerMetaInfo.key()), std::move(details)}};
}

struct Outcome {
    std::size_t              passed;
    gr::Size_t               failed;
    gr::Size_t               invalid;
    std::vector<std::string> errors;
};

[[nodiscard]] Outcome judge(std::vector<gr::property_map> results, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     script = fixture.emplace<EventScript>();
    script._events                   = std::move(results);
    auto& condition                  = fixture.emplace<MeasurementCondition>(std::move(settings));
    auto& tap                        = fixture.emplace<EventTap>();
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(script, condition).has_value());
    boost::ut::expect(fixture.connect<"evtOut", "evtIn">(condition, tap).has_value());
    gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
    boost::ut::expect(fixture.run().has_value());
    return {tap.size(), condition.n_failed.value, condition.n_invalid.value, sent.texts()};
}

const boost::ut::suite<"MeasurementCondition"> _measurementCondition = [] {
    using namespace boost::ut;
    const std::vector<gr::property_map> results{resultWith(3.), resultWith(7.), resultWith(std::nullopt), resultWith(5.)};

    "a result passes when its value meets the comparison"_test = [&] {
        const auto found = judge(results, {{"comparison", std::string("greater")}, {"lower", 4.}});
        expect(eq(found.passed, 2UZ)) << "7 and 5";
        expect(eq(found.failed, 1U)) << "3";
    };

    "an unknown comparison is refused: the previous one is kept and an error message is sent"_test = [&] {
        const auto found = judge(results, {{"comparison", std::string("sideways")}, {"lower", 4.}});
        expect(eq(found.passed, 2UZ)) << "7 and 5 under the kept greater";
        expect(eq(found.failed, 1U));
        expect(found.errors.size() == 1UZ && found.errors[0].contains("sideways"));
    };

    "an invalid result neither passes nor fails"_test = [&] {
        const auto found = judge(results, {{"comparison", std::string("greater")}, {"lower", 4.}});
        expect(eq(found.invalid, 1U));
    };

    "inside is the closed range, so its ends pass"_test = [&] {
        const auto found = judge(results, {{"comparison", std::string("inside")}, {"lower", 5.}, {"upper", 7.}});
        expect(eq(found.passed, 2UZ)) << "7 and 5, the range's own ends";
    };
};
} // namespace qa_measurement_condition

namespace qa_event_statistics {
using gr::trigger_test::EventScript;
using gr::trigger_test::EventTap;
using gr::trigger_test::kStreamStart;

[[nodiscard]] gr::property_map valueEvent(std::string name, std::uint64_t atNs, double value, bool inMeta = true) {
    gr::property_map event{{std::string(gr::tag::kTriggerName.key()), std::move(name)}, {std::string(gr::tag::kTriggerTime.key()), atNs}};
    if (inMeta) {
        event[std::string(gr::tag::kTriggerMetaInfo.key())] = gr::property_map{{"value", value}};
    } else {
        event[std::string("value")] = value;
    }
    return event;
}

template<typename T>
[[nodiscard]] T numberOr(const gr::property_map_view& map, std::string_view key, T fallback) {
    const T* found = map.get_if<T>(key);
    return found != nullptr ? *found : fallback;
}

struct Statistics {
    gr::Size_t            count = 0U;
    double                mean = 0., sdPopulation = 0., minimum = 0., maximum = 0., last = 0.;
    std::optional<double> sdSample;
    std::uint64_t         at = 0U;
};

[[nodiscard]] std::vector<Statistics> statisticsOf(const EventTap& tap) {
    std::vector<Statistics> found;
    for (const gr::property_map& event : tap._events) {
        const gr::property_map_view view{event};
        const auto                  meta = view.get_if<gr::property_map>(gr::tag::kTriggerMetaInfo.key());
        if (!meta) {
            continue;
        }
        const gr::property_map_view details{*meta};
        Statistics                  entry;
        entry.count        = numberOr<gr::Size_t>(details, std::string_view{"count"}, 0U);
        entry.mean         = numberOr<double>(details, std::string_view{"mean"}, -1.);
        entry.sdPopulation = numberOr<double>(details, std::string_view{"sd_population"}, -1.);
        entry.minimum      = numberOr<double>(details, std::string_view{"min"}, -1.);
        entry.maximum      = numberOr<double>(details, std::string_view{"max"}, -1.);
        entry.last         = numberOr<double>(details, std::string_view{"last"}, -1.);
        entry.sdSample     = details.get_if<double>(std::string_view{"sd_sample"}) != nullptr ? std::optional{*details.get_if<double>(std::string_view{"sd_sample"})} : std::nullopt;
        entry.at           = numberOr<std::uint64_t>(view, gr::tag::kTriggerTime.key(), 0U);
        found.push_back(entry);
    }
    return found;
}

[[nodiscard]] std::vector<Statistics> reduce(std::vector<gr::property_map> events, gr::property_map settings = {}) {
    gr::testing::GraphFixture fixture;
    auto&                     script = fixture.emplace<EventScript>();
    script._events                   = std::move(events);
    auto& block                      = fixture.emplace<gr::blocks::measurement::EventStatistics>(std::move(settings));
    auto& tap                        = fixture.emplace<EventTap>();
    boost::ut::expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, block, gr::PortDefinition{std::string("evtIn")}).has_value());
    boost::ut::expect(fixture.graph.connect(block, gr::PortDefinition{std::string("evtOut")}, tap, gr::PortDefinition{std::string("evtIn")}).has_value());
    boost::ut::expect(fixture.run().has_value());
    return statisticsOf(tap);
}

const boost::ut::suite<"EventStatistics (F9)"> _eventStatistics = [] {
    using namespace boost::ut;

    "the statistics of 2 4 4 4 5 5 7 9 end at mean 5, population sd 2, sample sd sqrt(32/7), min 2, max 9, last 9"_test = [] {
        std::vector<gr::property_map> events;
        std::uint64_t                 at = kStreamStart;
        for (const double value : {2., 4., 4., 4., 5., 5., 7., 9.}) {
            events.push_back(valueEvent("v", at, value));
            at += kMs;
        }
        const auto found = reduce(std::move(events));
        expect(eq(found.size(), 8UZ)) << fatal << "one report per value";
        const Statistics& end = found.back();
        expect(eq(end.count, 8U));
        expect(approx(end.mean, 5., 1e-12) and approx(end.sdPopulation, 2., 1e-12));
        expect(end.sdSample.has_value() && approx(*end.sdSample, std::sqrt(32. / 7.), 1e-12));
        expect(approx(end.minimum, 2., 1e-12) and approx(end.maximum, 9., 1e-12) and approx(end.last, 9., 1e-12));
        expect(eq(end.at, kStreamStart + 7U * kMs)) << "dated as the event that completed it";
    };

    "the first values give the running figures: one value has no sample sd, two values 3 and 1 and sqrt(2)"_test = [] {
        const auto found = reduce({valueEvent("v", kStreamStart, 2.), valueEvent("v", kStreamStart + kMs, 4.)});
        expect(eq(found.size(), 2UZ)) << fatal;
        expect(!found[0].sdSample.has_value() && approx(found[0].sdPopulation, 0., 1e-15) and approx(found[0].mean, 2., 1e-15));
        expect(approx(found[1].mean, 3., 1e-12) and approx(found[1].sdPopulation, 1., 1e-12));
        expect(found[1].sdSample.has_value() && approx(*found[1].sdSample, std::sqrt(2.), 1e-12));
    };

    "a 90-character trigger name still delivers every report with its root tuple"_test = [] {
        const std::string         name(90UZ, 'n');
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {valueEvent("v", kStreamStart, 2.), valueEvent("v", kStreamStart + kMs, 4.), valueEvent("v", kStreamStart + 2U * kMs, 9.)};
        auto& block                      = fixture.emplace<gr::blocks::measurement::EventStatistics>({{"trigger_name", name}});
        auto& tap                        = fixture.emplace<EventTap>();
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, block, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(block, gr::PortDefinition{std::string("evtOut")}, tap, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.run().has_value());
        expect(eq(tap._events.size(), 3UZ)) << fatal << "an event that does not fit degrades, it is not dropped";
        for (const gr::property_map& event : tap._events) {
            const gr::property_map_view view{event};
            expect(eq(std::string(view.value_or<std::string_view>(gr::tag::kTriggerName.key(), "")), name));
            expect(view.get_if<std::uint64_t>(gr::tag::kTriggerTime.key()) != nullptr);
        }
    };

    "a report larger than a slot degrades its meta from the end and still arrives with its root tuple"_test = [] {
        const std::string         field(700UZ, 'f');
        gr::testing::GraphFixture fixture;
        auto&                     script = fixture.emplace<EventScript>();
        auto                      event  = [&](std::uint64_t atNs, double value) {
            gr::property_map made{{std::string(gr::tag::kTriggerName.key()), std::string("v")}, {std::string(gr::tag::kTriggerTime.key()), atNs}};
            made[std::string(gr::tag::kTriggerMetaInfo.key())] = gr::property_map{{field, value}};
            return made;
        };
        script._events = {event(kStreamStart, 2.), event(kStreamStart + kMs, 4.), event(kStreamStart + 2U * kMs, 9.)};
        auto& block    = fixture.emplace<gr::blocks::measurement::EventStatistics>({{"field", field}});
        auto& tap      = fixture.emplace<EventTap>();
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, block, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(block, gr::PortDefinition{std::string("evtOut")}, tap, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.run().has_value());
        expect(eq(tap._events.size(), 3UZ)) << fatal << "a report that does not fit degrades, it is not dropped";
        for (std::size_t i = 0UZ; i < tap._events.size(); ++i) {
            const gr::property_map_view view{tap._events[i]};
            expect(eq(std::string(view.value_or<std::string_view>(gr::tag::kTriggerName.key(), "")), std::string(block.trigger_name.value)));
            expect(eq(numberOr<std::uint64_t>(view, gr::tag::kTriggerTime.key(), 0U), kStreamStart + i * kMs)) << "dated as the event that caused it";
        }
    };

    "a large offset does not cancel the deviation"_test = [] {
        const auto found = reduce({valueEvent("v", kStreamStart, 1e9 + 1.), valueEvent("v", kStreamStart + kMs, 1e9 + 3.)});
        expect(eq(found.size(), 2UZ)) << fatal;
        expect(approx(found[1].sdPopulation, 1., 1e-6));
    };

    "only events matching the filter and carrying the field count, wherever the field sits"_test = [] {
        const auto found = reduce(
            {
                valueEvent("v", kStreamStart, 1.),
                valueEvent("other", kStreamStart + kMs, 100.),
                valueEvent("v", kStreamStart + 2U * kMs, 3., false),
                gr::trigger_test::eventNamed("v", kStreamStart + 3U * kMs),
            },
            {{"filter", std::string("v")}});
        expect(eq(found.size(), 2UZ)) << fatal << "the other name is filtered out and the field-less event is skipped";
        expect(approx(found[1].mean, 2., 1e-12) and eq(found[1].count, 2U));
    };

    "per-gate means of a stream compose into their statistics"_test = [] {
        gr::testing::GraphFixture fixture;
        const std::vector<float>  samples{1.f, 1.f, 1.f, 1.f, 3.f, 3.f, 3.f, 3.f, 5.f, 5.f, 5.f, 5.f};
        auto&                     source = fixture.emplace<gr::testing::TagSource<float>>({
            {"values", samples},
            {"n_samples_max", static_cast<gr::Size_t>(samples.size())},
            {"mark_tag", false},
        });
        source._tags.push_back({0UZ, gr::property_map{{std::string(gr::tag::kTriggerTime.key()), kStreamStart}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}}});
        auto& gates   = fixture.emplace<gr::blocks::measurement::Measure<float, gr::blocks::measurement::Quantity::mean>>({{"length", 4U}, {"sample_rate", 1000.f}});
        auto& sink    = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        auto& reducer = fixture.emplace<gr::blocks::measurement::EventStatistics>();
        auto& tap     = fixture.emplace<EventTap>();
        expect(fixture.graph.connect(source, gr::PortDefinition{std::string("out")}, gates, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(gates, gr::PortDefinition{std::string("out#0")}, sink, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.graph.connect(gates, gr::PortDefinition{std::string("evtOut")}, reducer, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(reducer, gr::PortDefinition{std::string("evtOut")}, tap, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.run().has_value());
        const auto found = statisticsOf(tap);
        expect(eq(found.size(), 3UZ)) << fatal << "one report per gate";
        expect(eq(found.back().count, 3U) and approx(found.back().mean, 3., 1e-12) and approx(found.back().sdPopulation, std::sqrt(8. / 3.), 1e-12));
        expect(found.back().sdSample.has_value() && approx(*found.back().sdSample, 2., 1e-12));
    };
};
} // namespace qa_event_statistics

static_assert(gr::device::DeviceRelocatable<gr::blocks::measurement::MeasurementCondition> && gr::device::DeviceRelocatable<gr::blocks::measurement::EventStatistics>);

int main() {
    using namespace boost::ut;
    using gr::testing::operator""_domain_test;
    using gr::trigger_test::eventNamed;
    using gr::trigger_test::EventScript;

    // greater than 4: 3 fails, 7 and 5 pass, the result without a value is neither; dated as the results, in time order
    "a measurement condition on a device passes the results that meet it, in time order"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed = domain == "host" ? std::string("host") : std::string(domain) + "!";
        const auto        result = [](std::uint64_t at, std::optional<double> value) {
            gr::property_map event = eventNamed("result", at);
            if (value) {
                event[std::string(gr::tag::kTriggerMetaInfo.key())] = gr::property_map{{"value", *value}};
            }
            return event;
        };
        auto& script    = fixture.template emplace<EventScript>();
        script._events  = {result(1'000U, 3.), result(4'000U, 5.), result(2'000U, 7.), result(3'000U, std::nullopt)};
        auto& condition = fixture.template emplace<gr::blocks::measurement::MeasurementCondition>({{"comparison", std::string("greater")}, {"lower", 4.}, {"gr:compute_domain", placed}});
        auto& tap       = fixture.template emplace<gr::trigger_test::EventTap>();
        expect(fixture.template connect<"evtOut", "evtIn">(script, condition).has_value());
        expect(fixture.template connect<"evtOut", "evtIn">(condition, tap).has_value());
        expect(fixture.run().has_value()) << domain;
        const std::vector<std::pair<std::uint64_t, std::string>> expected{{2'000U, "result"}, {4'000U, "result"}};
        expect(tap.dated() == expected) << domain << ": 7 at 2 us and 5 at 4 us";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    // values 2, 4, 9: count 3, mean 5, min 2, max 9, last 9; squared deviations 9 + 1 + 16 = 26,
    // sd_population = sqrt(26 / 3) = 2.943920288775949, sd_sample = sqrt(26 / 2) = 3.605551275463989;
    // the fourth event lacks the field and is skipped: three reports, dated as their events
    "event statistics on a device accumulate a field by Welford's method"_domain_test = [&](auto& fixture, std::string_view domain) {
        const std::string placed   = domain == "host" ? std::string("host") : std::string(domain) + "!";
        const auto        measured = [](std::uint64_t at, double value) {
            gr::property_map event                              = eventNamed("m", at);
            event[std::string(gr::tag::kTriggerMetaInfo.key())] = gr::property_map{{"value", value}};
            return event;
        };
        auto& script   = fixture.template emplace<EventScript>();
        script._events = {measured(1'000U, 2.), measured(2'000U, 4.), measured(3'000U, 9.), eventNamed("m", 4'000U)};
        auto& stats    = fixture.template emplace<gr::blocks::measurement::EventStatistics>({{"filter", std::string("m")}, {"trigger_name", std::string("stats")}, {"gr:compute_domain", placed}});
        auto& tap      = fixture.template emplace<gr::trigger_test::EventTap>();
        expect(fixture.template connect<"evtOut", "evtIn">(script, stats).has_value());
        expect(fixture.template connect<"evtOut", "evtIn">(stats, tap).has_value());
        expect(fixture.run().has_value()) << domain;
        const std::vector<std::pair<std::uint64_t, std::string>> expected{{1'000U, "stats"}, {2'000U, "stats"}, {3'000U, "stats"}};
        expect(tap.dated() == expected) << domain;
        if (tap.size() == 3UZ) {
            const gr::property_map      meta = tap.metaOf(2);
            const gr::property_map_view last{meta};
            const auto                  number = [&](std::string_view key) {
                const auto* found = last.get_if<double>(key);
                return found ? *found : -1.;
            };
            const auto* count = last.get_if<gr::Size_t>(std::string_view{"count"});
            expect(count != nullptr && *count == 3U) << domain;
            expect(approx(number("mean"), 5., 1e-12)) << domain;
            expect(approx(number("min"), 2., 1e-12) && approx(number("max"), 9., 1e-12) && approx(number("last"), 9., 1e-12)) << domain;
            expect(approx(number("sd_population"), 2.943920288775949, 1e-12)) << domain;
            expect(approx(number("sd_sample"), 3.605551275463989, 1e-12)) << domain;
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
