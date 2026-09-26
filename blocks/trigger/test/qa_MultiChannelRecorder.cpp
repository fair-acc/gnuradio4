#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/Marble.hpp>
#include <gnuradio-4.0/trigger/MultiChannelRecorder.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::CollectingSink;
using gr::trigger_test::eventNamed;
using gr::trigger_test::EventScript;

namespace {
constexpr std::uint64_t kTriggerTime = 1'000'000'000U; // the tag dates sample 2 of both channels
constexpr float         kSampleRate  = 1000.f;

[[nodiscard]] gr::property_map tagsFor() {
    return gr::property_map{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, //
                                      {std::string(gr::tag::TRIGGER_TIME.key()), kTriggerTime},                      //
                                      {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f},                             //
                                      {std::string("sample_rate"), kSampleRate}}}};
}

} // namespace

const boost::ut::suite<"MultiChannelRecorder"> _recorder = [] {
    using namespace boost::ut;

    const gr::property_map lowValues{{"a", 1.f}, {"b", 2.f}, {"c", 3.f}, {"d", 4.f}, {"e", 5.f}, {"f", 6.f}, {"g", 7.f}};
    const gr::property_map highValues{{"a", 11.f}, {"b", 12.f}, {"c", 13.f}, {"d", 14.f}, {"e", 15.f}, {"f", 16.f}, {"g", 17.f}};

    "one decision extracts the same window from every channel"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     low    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f g |")}, {"sample_values", lowValues}, {"sample_tags", tagsFor()}});
        auto&                     high   = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f g |")}, {"sample_values", highValues}, {"sample_tags", tagsFor()}});
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {eventNamed("coincidence", kTriggerTime)};
        script._stopWhenDone             = false; // the streams decide when the graph ends, or the recorder's windows never fill
        auto& recorder                   = fixture.emplace<MultiChannelRecorder<float>>({{"n_inputs", 2U}, {"n_pre", 2U}, {"n_post", 2U}, {"sample_rate", kSampleRate}});
        auto& sets                       = fixture.emplace<CollectingSink<gr::DataSet<float>>>();

        expect(fixture.graph.connect(low, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(high, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#1")}).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, recorder, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("out")}, sets, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        expect(eq(recorder.n_recorded.value, 1U)) << "one decision, one segment";
        expect(eq(sets._collected.size(), 1UZ));
        if (sets._collected.size() == 1UZ) {
            const gr::DataSet<float>& set = sets._collected.front();
            expect(eq(set.signal_names.size(), 2UZ)) << "both channels in one set, so their extents match by construction";
            expect(eq(static_cast<std::size_t>(set.extents.at(0)), 5UZ)) << "two before the decision, the decision, two after";
            expect(eq(set.signal_values.size(), 10UZ));
            if (set.signal_values.size() == 10UZ) {
                expect(eq(set.signal_values[0], 1.f)) << "the window reaches back before the trigger";
                expect(eq(set.signal_values[2], 3.f)) << "and the trigger's own sample sits in the middle";
                expect(eq(set.signal_values[5], 11.f)) << "the second channel follows, same window";
                expect(eq(set.signal_values[7], 13.f));
            }

            gr::testing::MarbleDiagram diagram{"MultiChannelRecorder: one decision, one window from each channel"};
            diagram.row("ch0 tag").at(kTriggerTime, "start");
            diagram.row("ch1 tag").at(kTriggerTime, "start");
            diagram.row("decision").at(kTriggerTime, "coincidence");
            diagram.row("segment").at(static_cast<std::uint64_t>(set.timestamp), "DataSet");
            diagram.print();

            std::vector<double> x(static_cast<std::size_t>(set.extents.at(0)));
            std::vector<double> ch0(x.size());
            std::vector<double> ch1(x.size());
            for (std::size_t i = 0UZ; i < x.size(); ++i) {
                x[i]   = static_cast<double>(set.axis_values[0][i]);
                ch0[i] = static_cast<double>(set.signal_values[i]);
                ch1[i] = static_cast<double>(set.signal_values[x.size() + i]);
            }
            auto chart = gr::graphs::ImChart<80, 16>({{x.front(), x.back()}, {0., 20.}});
            chart.draw(x, ch0, "ch0");
            chart.draw(x, ch1, "ch1");
            chart.draw();
        }
    };

    "a decision that carries no time cannot place a window"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     low    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e |")}, {"sample_values", lowValues}, {"sample_tags", tagsFor()}});
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("coincidence")}}};
        script._stopWhenDone             = false; // the streams decide when the graph ends, or the recorder's windows never fill
        auto& recorder                   = fixture.emplace<MultiChannelRecorder<float>>({{"n_inputs", 1U}, {"n_pre", 1U}, {"n_post", 1U}, {"sample_rate", kSampleRate}});
        auto& sets                       = fixture.emplace<CollectingSink<gr::DataSet<float>>>();

        expect(fixture.graph.connect(low, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, recorder, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("out")}, sets, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        expect(eq(recorder.n_recorded.value, 0U));
        expect(eq(recorder.n_undated.value, 1U)) << "a window has to be placed somewhere, and nothing said where";
    };

    "n_segments stops the recorder after the count it was given"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     low    = fixture.emplace<MarbleSource<float>>({{"script", std::string("a b T:c d e f g |")}, {"sample_values", lowValues}, {"sample_tags", tagsFor()}});
        auto&                     script = fixture.emplace<EventScript>();
        script._events                   = {eventNamed("coincidence", kTriggerTime), eventNamed("coincidence", kTriggerTime + 1'000'000U)};
        script._stopWhenDone             = false; // the streams decide when the graph ends, or the recorder's windows never fill
        auto& recorder                   = fixture.emplace<MultiChannelRecorder<float>>({{"n_inputs", 1U}, {"n_pre", 1U}, {"n_post", 1U}, {"sample_rate", kSampleRate}, {"n_segments", 1U}});
        auto& sets                       = fixture.emplace<CollectingSink<gr::DataSet<float>>>();

        expect(fixture.graph.connect(low, gr::PortDefinition{std::string("out")}, recorder, gr::PortDefinition{std::string("in#0")}).has_value());
        expect(fixture.graph.connect(script, gr::PortDefinition{std::string("evtOut")}, recorder, gr::PortDefinition{std::string("evtIn")}).has_value());
        expect(fixture.graph.connect(recorder, gr::PortDefinition{std::string("out")}, sets, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.run().has_value());

        expect(eq(recorder.n_recorded.value, 1U)) << "two decisions, one segment asked for";
    };
};

int main() { /* tests are statically executed */ }
