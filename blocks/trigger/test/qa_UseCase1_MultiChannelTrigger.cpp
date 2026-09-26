#include <boost/ut.hpp>

#include <cmath>
#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/Coincidence.hpp>
#include <gnuradio-4.0/trigger/MultiChannelRecorder.hpp>
#include <gnuradio-4.0/trigger/ValueTrigger.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::CollectingSink;
using gr::trigger_test::EventTap;

namespace {
constexpr float         kSampleRate = 1000.f;         // 1 kHz, so one sample is one millisecond
constexpr std::uint64_t kStartTime  = 1'000'000'000U; // the tag that dates sample 0
constexpr std::size_t   kSamples    = 320UZ;
constexpr std::size_t   kChannels   = 4UZ;
constexpr float         kBandHeight = 7.f; // the vertical room each channel is given, so four traces are legible at once

/// a semi-periodic pulse response: a fast rise then an exponential decay, restarting every `period` samples with a
/// deterministic wobble, so the channels are neither aligned nor exactly periodic
struct PulseSource : gr::Block<PulseSource> {
    gr::PortOut<float> out;

    GR_MAKE_REFLECTABLE(PulseSource, out);

    std::size_t   _period     = 80UZ;
    std::size_t   _offset     = 0UZ;
    std::size_t   _wobble     = 0UZ;
    float         _height     = 5.f;
    float         _baseline   = 0.f; // each channel sits in its own band, as it would on a scope screen
    std::size_t   _nSamples   = kSamples;
    std::uint64_t _startTime  = kStartTime;
    float         _sampleRate = kSampleRate;
    std::size_t   _index      = 0UZ;

    /// where each pulse begins: every other one arrives `_wobble` samples late, so nothing is exactly periodic
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
            for (const std::size_t start : where) { // a fast rise, then an exponential decay
                if (at >= start && at - start < 16UZ) {
                    outSpan[i] = _baseline + _height * std::exp(-static_cast<float>(at - start) / 5.f);
                    break;
                }
            }
        }
        if (_index == 0UZ && n > 0UZ) {                                                                               // date the stream, so a trigger on it can say when an edge happened
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
} // namespace

const boost::ut::suite<"use case #1: per-channel triggers, one coincidence, one snapshot"> _useCase1 = [] {
    using namespace boost::ut;

    "four channels trigger independently, and one coincidence extracts a snapshot of all of them"_test = [] {
        constexpr std::array<std::size_t, kChannels> kPeriods{80UZ, 80UZ, 80UZ, 40UZ}; // channel 3 runs at twice the rate
        constexpr std::array<std::size_t, kChannels> kOffsets{0UZ, 3UZ, 7UZ, 2UZ};     // and each starts at its own time
        constexpr gr::Size_t                         kPre  = 8U;
        constexpr gr::Size_t                         kPost = 24U;

        gr::testing::GraphFixture fixture;
        auto&                     coincidence  = fixture.emplace<Coincidence>({{"filters", std::vector<std::string>{"ch0", "ch1", "ch2", "ch3"}}, //
                                 {"logic", std::string("all")},                                                                                   //
                                 {"window", 0.012},                                                                                               // 12 ms covers the spread of the starts
                                 {"holdoff", 0.050},                                                                                              // one report per pulse group
                                 {"trigger_name", std::string("all channels")}});
        auto&                     recorder     = fixture.emplace<MultiChannelRecorder<float>>({{"n_inputs", static_cast<gr::Size_t>(kChannels)}, //
                                    {"n_pre", kPre},                                                                                             //
                                    {"n_post", kPost},                                                                                           //
                                    {"history_margin", 256U},                                                                                    //
                                    {"sample_rate", kSampleRate}});
        auto&                     sets         = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        auto&                     triggerLog   = fixture.emplace<EventTap>();
        auto&                     compositeLog = fixture.emplace<EventTap>();

        std::vector<CollectingSink<float>*> streams;
        for (std::size_t channel = 0UZ; channel < kChannels; ++channel) {
            auto& source     = fixture.emplace<PulseSource>();
            source._period   = kPeriods[channel];
            source._offset   = kOffsets[channel];
            source._wobble   = channel + 1UZ;                                                                                       // a different wobble each, so nothing is exactly periodic
            source._baseline = static_cast<float>(channel) * kBandHeight;                                                           // stacked, so four traces read at once
            auto& trigger    = fixture.emplace<ValueTrigger<float, ValueCondition::level>>({{"threshold", source._baseline + 2.5f}, // each channel has its own level
                   {"hysteresis", 0.5f},                                                                                            //
                   {"sample_rate", kSampleRate},                                                                                    //
                   {"trigger_name", std::format("ch{}", channel)}});
            // an event output grants one slot per work call, so a call must not span two edges; the pulses are 40
            // samples apart at the closest, and the recorder is paced the same way so it does not outrun the events
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
            port.max_samples = 16UZ; // in step with the triggers, so the decisions arrive while the streams are still running
        }

        expect(fixture.run().has_value());

        // --- what happened, in numbers ---
        expect(gt(triggerLog.size(), 4UZ)) << "every channel triggers, repeatedly";
        expect(ge(coincidence.n_coincidences.value, 1U)) << "and the four of them fall inside one window";
        expect(ge(recorder.n_recorded.value, 1U)) << "each coincidence extracts one snapshot of every channel";
        expect(eq(recorder.n_undated.value, 0U)) << "the composite is dated, so the window can be placed";

        if (!sets._collected.empty()) {
            const gr::DataSet<float>& snapshot = sets._collected.front();
            expect(eq(snapshot.signal_names.size(), kChannels)) << "one signal per channel, in one set";
            expect(eq(static_cast<gr::Size_t>(snapshot.extents.at(0)), kPre + kPost + 1U)) << "the window asked for";
            expect(eq(snapshot.signal_values.size(), kChannels * static_cast<std::size_t>(snapshot.extents.at(0))));
        }

        // --- what happened, drawn ---
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
        auto& together = diagram.row("all four"); // the same triggers on one line, where they stack
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
            const gr::DataSet<float>& snapshot = sets._collected.front();
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

int main() { /* tests are statically executed */ }
