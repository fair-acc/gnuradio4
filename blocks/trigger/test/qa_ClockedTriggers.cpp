#include <boost/ut.hpp>

#include <string>
#include <vector>

#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/ClockedTriggers.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::testing::TagSource;
using gr::trigger_test::EventTap;

namespace {
/// a square clock, one sample low and one high, so an edge sits on every second sample
[[nodiscard]] std::vector<float> clockOf(std::size_t nEdges) {
    std::vector<float> wave;
    for (std::size_t i = 0UZ; i < nEdges; ++i) {
        wave.push_back(0.f);
        wave.push_back(5.f);
    }
    return wave;
}

template<typename TBlock>
[[nodiscard]] std::size_t runWith(std::vector<float> data, std::vector<float> clock, gr::property_map settings, TBlock** kept = nullptr) {
    gr::testing::GraphFixture fixture;
    const gr::Size_t          nSamples = static_cast<gr::Size_t>(std::min(data.size(), clock.size()));
    auto&                     dataSrc  = fixture.template emplace<TagSource<float>>({{"n_samples_max", nSamples}, {"values", std::move(data)}, {"mark_tag", false}});
    auto&                     clkSrc   = fixture.template emplace<TagSource<float>>({{"n_samples_max", nSamples}, {"values", std::move(clock)}, {"mark_tag", false}});
    auto&                     block    = fixture.template emplace<TBlock>(std::move(settings));
    auto&                     sink     = fixture.template emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
    auto&                     tap      = fixture.template emplace<EventTap>();

    boost::ut::expect(fixture.graph.connect(dataSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.graph.connect(clkSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("clk_in")}).has_value());
    boost::ut::expect(fixture.template connect<"out", "in">(block, sink).has_value());
    boost::ut::expect(fixture.template connect<"evtOut", "evtIn">(block, tap).has_value());
    boost::ut::expect(fixture.run().has_value());

    if (kept != nullptr) {
        *kept = std::addressof(block);
    }
    return tap.size();
}
} // namespace

const boost::ut::suite<"SerialPatternTrigger"> _serial = [] {
    using namespace boost::ut;

    const gr::property_map settings{{"threshold", 2.5f}, {"hysteresis", 1.f}, {"clock_threshold", 2.5f}, {"clock_edge", std::string("rising")}, {"sample_rate", 1000.f}};

    "the word the clock spelled out is found"_test = [&] {
        // data holds 1 0 1 1 across four clock edges, each edge on the odd samples
        const std::vector<float> data{5.f, 5.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f};
        gr::property_map         withPattern = settings;
        withPattern[std::string("pattern")]  = std::string("1011");

        SerialPatternTrigger<float>* block = nullptr;
        const std::size_t            found = runWith<SerialPatternTrigger<float>>(data, clockOf(4UZ), withPattern, &block);

        expect(eq(found, 1UZ)) << "the four bits spell the pattern exactly once";
        expect(block != nullptr && block->n_bits.value == 4U) << "one bit per clock edge";
    };

    "a word that does not match is not reported"_test = [&] {
        const std::vector<float> data{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        gr::property_map         withPattern = settings;
        withPattern[std::string("pattern")]  = std::string("1011");

        expect(eq(runWith<SerialPatternTrigger<float>>(data, clockOf(4UZ), withPattern), 0UZ));
    };

    "an X accepts whichever bit arrives"_test = [&] {
        const std::vector<float> data{5.f, 5.f, 0.f, 0.f, 5.f, 5.f, 0.f, 0.f};
        gr::property_map         withPattern = settings;
        withPattern[std::string("pattern")]  = std::string("1X1X");

        expect(eq(runWith<SerialPatternTrigger<float>>(data, clockOf(4UZ), withPattern), 1UZ)) << "1?1? matches 1 0 1 0";
    };

    "the data line passes through untouched"_test = [&] {
        const std::vector<float> data{5.f, 5.f, 0.f, 0.f};
        gr::property_map         withPattern = settings;
        withPattern[std::string("pattern")]  = std::string("10");

        gr::testing::GraphFixture fixture;
        auto&                     dataSrc = fixture.emplace<TagSource<float>>({{"n_samples_max", 4U}, {"values", data}, {"mark_tag", false}});
        auto&                     clkSrc  = fixture.emplace<TagSource<float>>({{"n_samples_max", 4U}, {"values", clockOf(2UZ)}, {"mark_tag", false}});
        auto&                     block   = fixture.emplace<SerialPatternTrigger<float>>(withPattern);
        auto&                     sink    = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.graph.connect(dataSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.graph.connect(clkSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("clk_in")}).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(sink._samples.size(), 4UZ)) << "what is triggered on is the word, not the signal";
    };
};

const boost::ut::suite<"SetupHoldTrigger"> _setupHold = [] {
    using namespace boost::ut;

    const gr::property_map settings{{"threshold", 2.5f}, {"hysteresis", 1.f}, {"clock_threshold", 2.5f}, {"clock_edge", std::string("rising")}, //
        {"setup_samples", 2U}, {"hold_samples", 2U}, {"sample_rate", 1000.f}};

    "data settled well before the edge is no violation"_test = [&] {
        // the data rises at sample 0 and stays; the clock edges are at 1, 3, 5 ...
        const std::vector<float> data{5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f};
        SetupHoldTrigger<float>* block = nullptr;

        expect(eq(runWith<SetupHoldTrigger<float>>(data, clockOf(5UZ), settings, &block), 0UZ));
        expect(block != nullptr && block->n_checks.value == 5U) << "every edge was judged";
    };

    "data that moves inside the window is a violation"_test = [&] {
        // the data changes at sample 4, which is inside the window of the edge at 5 and of the edge at 3
        const std::vector<float> data{0.f, 0.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f};
        SetupHoldTrigger<float>* block = nullptr;

        const std::size_t found = runWith<SetupHoldTrigger<float>>(data, clockOf(5UZ), settings, &block);
        expect(ge(found, 1UZ)) << "the change sits in the setup half of one edge and the hold half of another";
        expect(block != nullptr && block->n_violations.value == found);
    };

    "a violation is reported once its hold half has arrived, not before"_test = [&] {
        const std::vector<float> data{0.f, 0.f, 0.f, 0.f, 5.f, 5.f};
        SetupHoldTrigger<float>* block = nullptr;

        const std::size_t found = runWith<SetupHoldTrigger<float>>(data, clockOf(3UZ), settings, &block);
        expect(block != nullptr) << "the block outlives the run";
        expect(le(found, block->n_checks.value)) << "an edge whose hold samples never arrived is not judged at all";
    };

    "the window and the violation in it, drawn"_test = [&] {
        const std::vector<float>  data{0.f, 0.f, 0.f, 0.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f};
        gr::testing::GraphFixture fixture;
        auto&                     dataSrc = fixture.emplace<TagSource<float>>({{"n_samples_max", 10U}, {"values", data}, {"mark_tag", false}});
        auto&                     clkSrc  = fixture.emplace<TagSource<float>>({{"n_samples_max", 10U}, {"values", clockOf(5UZ)}, {"mark_tag", false}});
        auto&                     block   = fixture.emplace<SetupHoldTrigger<float>>(settings);
        auto&                     tap     = fixture.emplace<EventTap>();
        expect(fixture.graph.connect(dataSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("in")}).has_value());
        expect(fixture.graph.connect(clkSrc, gr::PortDefinition{std::string("out")}, block, gr::PortDefinition{std::string("clk_in")}).has_value());
        expect(fixture.connect<"evtOut", "evtIn">(block, tap).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"SetupHoldTrigger: the data moves at sample 4, inside two clock windows"};
        diagram.unit   = "sample";
        auto& clockRow = diagram.row("clk");
        for (std::uint64_t at = 1U; at < 10U; at += 2U) {
            clockRow.at(at, "edge");
        }
        clockRow.completes();
        diagram.row("in").at(4U, "data changes").completes();
        diagram.condition(std::format("SetupHoldTrigger(setup = 2, hold = 2), {} of {} edges violated", block.n_violations.value, block.n_checks.value));
        auto& out = diagram.row("evtOut");
        for (std::size_t i = 0UZ; i < tap.size(); ++i) {
            out.at(3U + 2U * i, "violation");
        }
        out.completes();
        diagram.print();

        expect(ge(block.n_violations.value, 1U));
    };
};

int main() { /* tests are statically executed */ }
