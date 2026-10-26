#include <boost/ut.hpp>

#include <cmath>
#include <format>
#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/Coincidence.hpp>
#include <gnuradio-4.0/trigger/MultiChannelRecorder.hpp>
#include <gnuradio-4.0/trigger/SchmittTrigger.hpp>
#include <gnuradio-4.0/trigger/ValueTrigger.hpp>

#include "Marble.hpp"
#include "TriggerTest.hpp"

namespace qa_use_case_1 {
using gr::trigger_test::kSampleRate;
using gr::trigger_test::kStreamStart;
using namespace gr::blocks::trigger;
using gr::trigger_test::CollectingSink;
using gr::trigger_test::EventTap;

constexpr std::size_t kSamples    = 320UZ;
constexpr std::size_t kChannels   = 4UZ;
constexpr float       kBandHeight = 7.f;

/**
 * a semi-periodic pulse response: a fast rise then an exponential decay, restarting every `period` samples with a
 * deterministic wobble, so the channels are neither aligned nor exactly periodic
 */
struct PulseSource : gr::Block<PulseSource> {
    gr::PortOut<float> out;

    GR_MAKE_REFLECTABLE(PulseSource, out);

    std::size_t   _period     = 80UZ;
    std::size_t   _offset     = 0UZ;
    std::size_t   _wobble     = 0UZ;
    float         _height     = 5.f;
    float         _baseline   = 0.f;
    std::size_t   _nSamples   = kSamples;
    std::uint64_t _startTime  = kStreamStart;
    float         _sampleRate = kSampleRate;
    std::size_t   _index      = 0UZ;

    /** where each pulse begins: every other one arrives `_wobble` samples late, so nothing is exactly periodic */
    [[nodiscard]] std::vector<std::size_t> starts() const {
        std::vector<std::size_t> found;
        for (std::size_t pulse = 0UZ; _offset + pulse * _period < _nSamples; ++pulse) {
            found.push_back(_offset + pulse * _period + (pulse % 2UZ) * _wobble);
        }
        return found;
    }

    gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        const std::size_t              n     = std::min(outSpan.size(), _nSamples > _index ? _nSamples - _index : 0UZ);
        const std::vector<std::size_t> where = starts();
        for (std::size_t i = 0UZ; i < n; ++i) {
            const std::size_t at = _index + i;
            outSpan[i]           = _baseline;
            for (const std::size_t start : where) {
                if (at >= start && at - start < 16UZ) {
                    outSpan[i] = _baseline + _height * std::exp(-static_cast<float>(at - start) / 5.f);
                    break;
                }
            }
        }
        if (_index == 0UZ && n > 0UZ) {                                                                               //
            outSpan.publishTag(gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("acq_start")}, //
                                   {std::string(gr::tag::TRIGGER_TIME.key()), _startTime},                            //
                                   {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},                                 //
                                   {std::string("sample_rate"), _sampleRate}},
                0UZ);
        }
        _index += n;
        outSpan.publish(n);
        if (_index >= _nSamples) {
            this->requestStop();
            return gr::work::Status::DONE;
        }
        return gr::work::Status::OK;
    }
};

const boost::ut::suite<"use case #1: per-channel triggers, one coincidence, one snapshot"> _useCase1 = [] {
    using namespace boost::ut;

    "four channels trigger independently, and one coincidence extracts a snapshot of all of them"_test = [] {
        constexpr std::array<std::size_t, kChannels> kPeriods{80UZ, 80UZ, 80UZ, 40UZ};
        constexpr std::array<std::size_t, kChannels> kOffsets{0UZ, 3UZ, 7UZ, 2UZ};
        constexpr gr::Size_t                         kPre  = 8U;
        constexpr gr::Size_t                         kPost = 24U;

        gr::testing::GraphFixture fixture;
        auto&                     coincidence  = fixture.emplace<Coincidence>({{"filters", std::vector<std::string>{"ch0", "ch1", "ch2", "ch3"}}, //
                                 {"logic", std::string("all")},                                                                                   //
                                 {"window", 0.012},                                                                                               //
                                 {"holdoff", 0.050},                                                                                              //
                                 {"trigger_name", std::string("all channels")}});
        auto&                     recorder     = fixture.emplace<MultiChannelRecorder<float>>({{"n_inputs", static_cast<gr::Size_t>(kChannels)}, //
                                    {"n_pre", kPre},                                                                                             //
                                    {"n_post", kPost},                                                                                           //
                                    {"history", 0.3f},                                                                                           //
                                    {"sample_rate", kSampleRate}});
        auto&                     sets         = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        auto&                     triggerLog   = fixture.emplace<EventTap>();
        auto&                     compositeLog = fixture.emplace<EventTap>();

        std::vector<CollectingSink<float>*> streams;
        for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
            auto& source     = fixture.emplace<PulseSource>();
            source._period   = kPeriods[channel];
            source._offset   = kOffsets[channel];
            source._wobble   = channel + 1UZ;
            source._baseline = static_cast<float>(channel) * kBandHeight;
            auto& trigger    = fixture.emplace<ValueTrigger<float, ValueCondition::level>>({{"threshold", source._baseline + 2.5f}, //
                   {"hysteresis", 0.5f},                                                                                            //
                   {"sample_rate", kSampleRate},                                                                                    //
                   {"trigger_name", std::format("ch{}", channel)}});
            /**
             * an event output grants one slot per work call, so a call must not span two edges; the pulses are 40
             * samples apart at the closest, and the recorder is paced the same way so it does not outrun the events
             */
            trigger.in.max_samples = 16UZ;
            auto& stream           = fixture.emplace<CollectingSink<float>>();
            streams.push_back(&stream);

            expect(fixture.connect<"out", "in">(source, trigger).has_value()) << channel;
            expect(fixture.connect<"out", "in">(trigger, stream).has_value()) << channel;
            expect(fixture.graph.connect(source, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::format("in#{}", channel)}).has_value()) << channel;
            expect(fixture.connect<"evtOut", "evtIn">(trigger, coincidence).has_value()) << channel;
            expect(fixture.connect<"evtOut", "evtIn">(trigger, triggerLog).has_value()) << channel;
        }
        expect(fixture.graph.connect(coincidence, gr::PortDefinition{std::string("evtOut")}, recorder, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(coincidence, compositeLog).has_value());
        expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("out")}, sets, gr::PortDefinition{std::string("in")}).has_value());
        for (auto& port : recorder.in) {
            port.max_samples = 16UZ;
        }

        expect(fixture.run().has_value());

        expect(gt(triggerLog.size(), 4UZ)) << "every channel triggers, repeatedly";
        expect(ge(coincidence.n_coincidences.value, 1U)) << "and the four of them fall inside one window";
        expect(ge(recorder.n_recorded.value, 1U)) << "each coincidence extracts one snapshot of every channel";
        expect(eq(recorder.n_undated.value, 0U)) << "the composite is dated, so the window can be placed";

        if (!sets._collected.empty()) {
            const gr::DataSet<float>& snapshot = sets._collected.back();
            expect(eq(snapshot.signal_names.size(), kChannels)) << "one signal per channel, in one set";
            expect(eq(static_cast<gr::Size_t>(snapshot.extents.at(0)), kPre + kPost)) << "the window asked for";
            expect(eq(snapshot.signal_values.size(), kChannels * static_cast<std::size_t>(snapshot.extents.at(0))));
        }

        const auto triggerEvents = triggerLog.dated();

        gr::testing::MarbleDiagram diagram{"use case #1: four channels, one coincidence, one snapshot"};
        for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
            const std::string                wanted = std::format("ch{}", channel);
            gr::testing::MarbleDiagram::Row& row    = diagram.row(wanted);
            for (const auto& [at, name] : triggerEvents) {
                if (name == wanted) {
                    row.at(at, name);
                }
            }
        }
        auto& together = diagram.row("all four");
        for (const auto& [at, name] : triggerEvents) {
            together.at(at, name);
        }
        diagram.condition("Coincidence(all four, within 12 ms, 50 ms holdoff)");
        auto& composite = diagram.row("snapshot");
        for (const auto& [at, name] : compositeLog.dated()) {
            composite.at(at, name);
        }
        diagram.print();

        if (!streams.empty() && !streams[0]->_collected.empty()) {
            const std::size_t   shown = std::min(streams[0]->_collected.size(), 200UZ);
            std::vector<double> x(shown);
            for (std::size_t i = 0UZ; i < shown; ++i) {
                x[i] = static_cast<double>(i) / static_cast<double>(kSampleRate);
            }
            auto continuous = gr::graphs::ImChart<110, 26>({{x.front(), x.back()}, {-1., static_cast<double>(kChannels) * static_cast<double>(kBandHeight)}});
            for (std::size_t channel = 0UZ; channel < streams.size(); ++channel) {
                std::vector<double> y(shown);
                for (std::size_t i = 0UZ; i < shown; ++i) {
                    y[i] = static_cast<double>(streams[channel]->_collected[i]);
                }
                continuous.draw(x, y, std::format("ch{}", channel));
            }
            std::println("\nthe four continuous streams, as the triggers see them:");
            continuous.draw();
        }

        if (!sets._collected.empty()) {
            const gr::DataSet<float>& snapshot = sets._collected.back();
            const std::size_t         length   = static_cast<std::size_t>(snapshot.extents.at(0));
            std::vector<double>       x(length);
            for (std::size_t i = 0UZ; i < length; ++i) {
                x[i] = static_cast<double>(snapshot.axis_values[0][i]);
            }
            auto extracted = gr::graphs::ImChart<110, 26>({{x.front(), x.back()}, {-1., static_cast<double>(kChannels) * static_cast<double>(kBandHeight)}});
            for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
                std::vector<double> y(length);
                for (std::size_t i = 0UZ; i < length; ++i) {
                    y[i] = static_cast<double>(snapshot.signal_values[channel * length + i]);
                }
                extracted.draw(x, y, std::format("ch{}", channel));
            }
            std::println("\nthe extracted snapshot, zero at the coincidence ({} of {}):", 1, sets._collected.size());
            extracted.draw();
        }
    };
};
} // namespace qa_use_case_1

namespace qa_use_case_2 {
using gr::trigger_test::kStreamStart;
using namespace gr::blocks::trigger;
using gr::trigger_test::CollectingSink;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;

/**
 * a stream whose sample at index i has the value i, dated by a tag on its first sample, and optionally re-anchored by a
 * second dated tag -- the multi-rate fixture every test here builds on
 */
struct Channel {
    float                      rate;
    std::size_t                length;
    std::optional<std::size_t> reanchorAt{};
    std::uint64_t              reanchorTime = 0U;

    [[nodiscard]] std::string script() const {
        std::string out;
        for (std::size_t i = 0UZ; i < length; ++i) {
            out += i == 0UZ ? "T:" : (reanchorAt == i ? "U:" : "");
            out += std::format("s{} ", i);
        }
        return out + "|";
    }

    [[nodiscard]] gr::property_map values() const {
        gr::property_map map;
        for (std::size_t i = 0UZ; i < length; ++i) {
            map[std::format("s{}", i)] = static_cast<float>(i);
        }
        return map;
    }

    [[nodiscard]] gr::property_map tags() const {
        const auto dated = [this](std::uint64_t at) { return gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("anchor")}, {std::string(gr::tag::TRIGGER_TIME.key()), at}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}, {std::string("sample_rate"), rate}}; };
        return gr::property_map{{"T", dated(kStreamStart)}, {"U", dated(reanchorTime)}};
    }
};

struct Outcome {
    std::vector<gr::DataSet<float>> sets;
    gr::Size_t                      recorded;
    gr::Size_t                      unplaceable;
};

/** one recorder per channel, since their rates differ; every recorder hears the same decisions */
[[nodiscard]] std::vector<Outcome> recordAt(const std::vector<Channel>& channels, std::vector<gr::property_map> decisions, std::size_t chunk = 0UZ) {
    gr::testing::GraphFixture fixture;
    auto&                     script = fixture.emplace<EventScript>();
    script._events                   = std::move(decisions);
    script._stopWhenDone             = false;
    std::vector<MultiChannelRecorder<float>*>        recorders;
    std::vector<CollectingSink<gr::DataSet<float>>*> sinks;
    for (const Channel& channel : channels) {
        auto& source   = fixture.emplace<MarbleSource<float>>({{"script", channel.script()}, {"sample_values", channel.values()}, {"sample_tags", channel.tags()}});
        auto& recorder = fixture.emplace<MultiChannelRecorder<float>>({{"n_inputs", 1U}, {"trigger_pre", std::uint64_t{1'000'000U}}, {"trigger_post", std::uint64_t{1'000'000U}}, {"sample_rate", channel.rate}, {"history", 0.02f}});
        auto& sink     = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        boost::ut::expect(fixture.graph.connect(source, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#0")}).has_value());
        boost::ut::expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, recorder, gr::PortDefinition{std::string("evtIn")}).has_value());
        boost::ut::expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("out")}, sink, gr::PortDefinition{std::string("in")}).has_value());
        if (chunk > 0UZ) {
            recorder.in[0].max_samples = chunk;
        }
        recorders.push_back(&recorder);
        sinks.push_back(&sink);
    }
    boost::ut::expect(fixture.run().has_value());

    std::vector<Outcome> found;
    for (std::size_t i = 0UZ; i < recorders.size(); ++i) {
        found.push_back({sinks[i]->_collected, recorders[i]->n_recorded.value, recorders[i]->n_unplaceable.value});
    }
    return found;
}

[[nodiscard]] bool sameRecording(const std::vector<Outcome>& lhs, const std::vector<Outcome>& rhs) {
    const auto sameSet = [](const gr::DataSet<float>& a, const gr::DataSet<float>& b) { //
        return a.timestamp == b.timestamp && a.signal_values == b.signal_values && a.axis_values == b.axis_values && a.extents == b.extents && a.meta_information == b.meta_information;
    };
    return std::ranges::equal(lhs, rhs, [&](const Outcome& a, const Outcome& b) { return a.recorded == b.recorded && a.unplaceable == b.unplaceable && std::ranges::equal(a.sets, b.sets, sameSet); });
}

const boost::ut::suite<"use case #2: multi-rate acquisition on one clock, replayed"> _multiRate = [] {
    using namespace boost::ut;

    const std::vector<Channel> twoRates{{.rate = 1000.f, .length = 16UZ}, {.rate = 2000.f, .length = 32UZ}};

    "one decision cuts the same stretch of time from channels at different rates"_test = [&] {
        const auto found = recordAt(twoRates, {eventNamed("decision", kStreamStart + 3'000'000U)});
        expect(eq(found.size(), 2UZ)) << fatal;
        expect(eq(found[0].sets.size(), 1UZ) && eq(found[1].sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found[0].sets.front().signal_values, std::vector{2.f, 3.f})) << "1 kHz: 1 ms either side of sample 3";
        expect(std::ranges::equal(found[1].sets.front().signal_values, std::vector{4.f, 5.f, 6.f, 7.f})) << "2 kHz: the same 2 ms, twice the samples";
        expect(eq(found[0].sets.front().timestamp, found[1].sets.front().timestamp)) << "and both are dated alike";
    };

    "a later dated tag re-anchors the channel that carries it"_test = [&] {
        std::vector<Channel> skewed = twoRates;
        skewed[0].reanchorAt        = 5UZ;
        skewed[0].reanchorTime      = kStreamStart + 6'000'000U;
        const auto found            = recordAt(skewed, {eventNamed("decision", kStreamStart + 8'000'000U)});
        expect(eq(found[0].sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found[0].sets.front().signal_values, std::vector{6.f, 7.f})) << "placed from the newer anchor: sample 7, not 8";
        expect(eq(found[1].sets.size(), 1UZ)) << fatal;
        expect(std::ranges::equal(found[1].sets.front().signal_values,
            std::vector{
                14.f,
                15.f,
                16.f,
                17.f,
            }))
            << "the other channel counts on from its own anchor";
    };

    "a replay of the same streams and decisions records the same thing"_test = [&] {
        const std::vector<gr::property_map> decisions{eventNamed("first", kStreamStart + 3'000'000U), eventNamed("second", kStreamStart + 9'000'000U), eventNamed("too late", kStreamStart + 60'000'000U)};
        const auto                          first  = recordAt(twoRates, decisions);
        const auto                          replay = recordAt(twoRates, decisions);
        expect(eq(first[0].recorded, 2U));
        expect(sameRecording(first, replay)) << "data sets, time axes, meta information and counters";
        for (const std::size_t chunk : {1UZ, 2UZ, 5UZ}) {
            expect(sameRecording(first, recordAt(twoRates, decisions, chunk))) << "however the stream is cut:" << chunk;
        }
    };

    "an edge on one channel records another, and a replay records it again"_test = [] {
        const auto run = [] {
            gr::testing::GraphFixture fixture;
            const Channel             watched{.rate = 1000.f, .length = 16UZ};
            gr::property_map          stepValues = watched.values();
            for (std::size_t i = 8UZ; i < 16UZ; ++i) {
                stepValues[std::format("s{}", i)] = 10.f;
            }
            const Channel recorded{.rate = 2000.f, .length = 32UZ};
            auto&         step     = fixture.emplace<MarbleSource<float>>({{"script", watched.script()}, {"sample_values", stepValues}, {"sample_tags", watched.tags()}});
            auto&         trigger  = fixture.emplace<SchmittTrigger<float, gr::trigger::InterpolationMethod::NO_INTERPOLATION>>({
                {"threshold", 9.f},
                {"hysteresis", 2.f},
                {"sample_rate", 1000.f},
                {"edge", std::string("rising")},
            });
            auto&         spent    = fixture.emplace<MarbleSink<float>>({{"sample_values", stepValues}, {"sample_tags", watched.tags()}});
            auto&         stream   = fixture.emplace<MarbleSource<float>>({{"script", recorded.script()}, {"sample_values", recorded.values()}, {"sample_tags", recorded.tags()}});
            auto&         recorder = fixture.emplace<MultiChannelRecorder<float>>({{"n_inputs", 1U}, {"trigger_pre", std::uint64_t{1'000'000U}}, {"trigger_post", std::uint64_t{1'000'000U}}, {"sample_rate", 2000.f}, {"history", 0.02f}});
            auto&         sets     = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
            boost::ut::expect(fixture.connect<"out", "in">(step, trigger).has_value());
            boost::ut::expect(fixture.connect<"out", "in">(trigger, spent).has_value());
            boost::ut::expect(fixture.graph.connect(trigger, gr::PortDefinition{std::string("evtOut")}, recorder, gr::PortDefinition{std::string("evtIn")}).has_value());
            boost::ut::expect(fixture.graph.connect(stream, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#0")}).has_value());
            boost::ut::expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("out")}, sets, gr::PortDefinition{std::string("in")}).has_value());
            boost::ut::expect(fixture.run().has_value());
            return std::vector<Outcome>{{sets._collected, recorder.n_recorded.value, recorder.n_unplaceable.value}};
        };
        const auto first = run();
        expect(eq(first[0].recorded, 1U)) << fatal << "the edge at 8 ms opens one window on the other channel";
        expect(std::ranges::equal(first[0].sets.front().signal_values, std::vector{14.f, 15.f, 16.f, 17.f})) << "16 at 2 kHz is 8 ms";
        expect(sameRecording(first, run()));
    };
};
} // namespace qa_use_case_2

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
