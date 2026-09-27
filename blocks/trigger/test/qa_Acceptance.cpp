#include <boost/ut.hpp>

#include <string>

#include <gnuradio-4.0/trigger/EventReduce.hpp>
#include <gnuradio-4.0/trigger/EventShaping.hpp>
#include <gnuradio-4.0/trigger/Gate.hpp>
#include <gnuradio-4.0/trigger/Predicate.hpp>
#include <gnuradio-4.0/trigger/SampleAndHold.hpp>
#include <gnuradio-4.0/trigger/SchmittTrigger.hpp>
#include <gnuradio-4.0/trigger/StreamOps.hpp>
#include <gnuradio-4.0/trigger/TagBridge.hpp>
#include <gnuradio-4.0/trigger/TakeSkip.hpp>
#include <gnuradio-4.0/trigger/TriggerWatchdog.hpp>
#include <gnuradio-4.0/trigger/ValueTrigger.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::acceptsEveryTagPlacement;

const boost::ut::suite<"acceptance: the same answer however the stream is cut"> _chunkInvariance = [] {
    using namespace boost::ut;

    const gr::property_map onStart{{"filter", std::string("start")}};

    "Gate over each of its six modes"_test = [&] {
        for (const std::string mode : {"once", "toggle", "cooldown", "wait", "take_until", "skip_until"}) {
            acceptsEveryTagPlacement<Gate<float>>(std::format("Gate({})", mode), //
                {{"mode", mode}, {"open_filter", std::string("start")}, {"n_open", 2U}, {"n_delay", 2U}, {"n_cooldown", 2U}});
        }
    };

    "TakeN and SkipN"_test = [&] {
        acceptsEveryTagPlacement<TakeN<float>>("TakeN", {{"filter", std::string("start")}, {"n", 2U}});
        acceptsEveryTagPlacement<SkipN<float>>("SkipN", {{"filter", std::string("start")}, {"n", 2U}});
    };

    "SampleAndHold"_test = [&] { acceptsEveryTagPlacement<SampleAndHold<float>>("SampleAndHold", {{"filter", std::string("start")}, {"initial_value", 0.f}}); };

    "Debounce and Throttle"_test = [&] {
        acceptsEveryTagPlacement<Debounce<float>>("Debounce", {{"filter", std::string("start")}, {"n_samples", 2U}});
        acceptsEveryTagPlacement<Throttle<float>>("Throttle", {{"filter", std::string("start")}, {"n_samples", 2U}});
    };

    "the predicate three"_test = [&] {
        acceptsEveryTagPlacement<TakeWhile<float>>("TakeWhile", {{"predicate", std::string("less")}, {"threshold", 3.f}});
        acceptsEveryTagPlacement<SkipWhile<float>>("SkipWhile", {{"predicate", std::string("less")}, {"threshold", 3.f}});
        acceptsEveryTagPlacement<ElementAt<float>>("ElementAt", {{"n", 1U}});
        acceptsEveryTagPlacement<ElementAt<float>>("ElementAt(per segment)", {{"n", 1U}, {"segment_filter", std::string("start")}});
    };

    "Distinct and DistinctUntilChanged"_test = [&] {
        acceptsEveryTagPlacement<Distinct<float>>("Distinct", {{"max_values", 8U}});
        acceptsEveryTagPlacement<DistinctUntilChanged<float>>("DistinctUntilChanged", {});
    };

    "Accumulate over its cadence"_test = [&] {
        acceptsEveryTagPlacement<Accumulate<float, Accumulation::AUTO>>("Reduce(sum)", {{"operation", std::string("sum")}, {"n_samples", 2U}});
        acceptsEveryTagPlacement<Accumulate<float, Accumulation::last>>("Last(per segment)", {{"segment_filter", std::string("start")}, {"n_samples", 0U}});
    };

    "SampleFilter over each comparison"_test = [&] {
        for (const std::string predicate : {"greater", "greater_equal", "less", "less_equal", "equal", "not_equal"}) {
            acceptsEveryTagPlacement<SampleFilter<float>>(std::format("SampleFilter({})", predicate), {{"predicate", predicate}, {"threshold", 1.}});
        }
    };

    "Scan, with and without a reset"_test = [&] {
        acceptsEveryTagPlacement<Scan<float>>("Scan(sum)", {{"operation", std::string("sum")}});
        acceptsEveryTagPlacement<Scan<float>>("Scan(mean, reset)", {{"operation", std::string("mean")}, {"reset_filter", std::string("start")}});
    };

    "TakeLast and SkipLast"_test = [&] {
        acceptsEveryTagPlacement<TakeLast<float>>("TakeLast", {{"filter", std::string("start")}, {"n", 2U}});
        acceptsEveryTagPlacement<SkipLast<float>>("SkipLast", {{"filter", std::string("start")}, {"n", 2U}});
        acceptsEveryTagPlacement<TakeLast<float>>("TakeLast(whole stream)", {{"n", 2U}});
    };

    "Repeat"_test = [&] { acceptsEveryTagPlacement<Repeat<float>>("Repeat", {{"n_repeats", 2U}, {"capacity", 8U}, {"segment_filter", std::string("start")}}); };

    "the blocks that pass the stream through untouched"_test = [&] {
        acceptsEveryTagPlacement<TagToMessage<float>>("TagToMessage", onStart);
        acceptsEveryTagPlacement<TriggerWatchdog<float>>("TriggerWatchdog", {{"filter", std::string("start")}, {"timeout_samples", 2U}});
        // an interpolating trigger declares its window through in.min_samples, so the harness only offers it spans it
        // admits; qa_SchmittTriggerBlock holds it to the same edges over a script long enough for several of them
        acceptsEveryTagPlacement<SchmittTrigger<float, gr::trigger::InterpolationMethod::BASIC_LINEAR_INTERPOLATION>>("SchmittTrigger", {{"threshold", 2.f}, {"offset", 2.5f}});
        acceptsEveryTagPlacement<SchmittTrigger<float, gr::trigger::InterpolationMethod::NO_INTERPOLATION>>("SchmittTrigger(no interpolation)", {{"threshold", 2.f}, {"offset", 2.5f}});
        acceptsEveryTagPlacement<ValueTrigger<float, ValueCondition::level>>("LevelTrigger", {{"threshold", 2.5f}, {"hysteresis", 1.f}});
    };
};

const boost::ut::suite<"acceptance: what a block does when a port is not there"> _disconnected = [] {
    using namespace boost::ut;

    "a block whose event input is unconnected still runs"_test = [] {
        // every case above leaves evtIn unconnected already; this says so on purpose, for the blocks that read one
        const auto gated = gr::trigger_test::scriptsAtChunkSizes<Gate<float>>("T:a b c d |", {{"mode", std::string("once")}, {"open_filter", std::string("start")}, {"n_open", 2U}});
        expect(!gated.empty() && !gated[0].empty()) << "an unconnected event bus is not an empty one that never ends";

        const auto debounced = gr::trigger_test::scriptsAtChunkSizes<Debounce<float>>("T:a b c d |", {{"filter", std::string("start")}, {"n_samples", 2U}});
        expect(!debounced.empty() && !debounced[0].empty());
    };

    "a block whose event output is unconnected still runs"_test = [] {
        const auto reported = gr::trigger_test::scriptsAtChunkSizes<TagToMessage<float>>("T:a b c |", {{"filter", std::string("start")}});
        expect(!reported.empty() && !reported[0].empty()) << "the events go nowhere, and that is not an error";
    };

    "the stream terminator reaches the sink in every case"_test = [] {
        const auto answers = gr::trigger_test::scriptsAtChunkSizes<SampleAndHold<float>>("T:a b c |", {{"filter", std::string("start")}});
        for (const std::string& answer : answers) {
            expect(answer.ends_with("|")) << std::format("'{}' does not say how the stream ended", answer);
        }
    };
};

const boost::ut::suite<"acceptance: a settings change between runs"> _settings = [] {
    using namespace boost::ut;

    "a changed setting takes effect and leaves no state behind"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     block = fixture.emplace<TakeN<float>>({{"filter", std::string("start")}, {"n", 2U}});
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();
        expect(eq(block.n.value, 2U));

        expect(block.settings().set({{"n", gr::Size_t(3)}}).empty());
        std::ignore = block.settings().activateContext();
        std::ignore = block.settings().applyStagedParameters();
        expect(eq(block.n.value, 3U)) << "the new setting is the one that applies";

        const auto two   = gr::trigger_test::scriptsAtChunkSizes<TakeN<float>>("a T:b c d e |", {{"filter", std::string("start")}, {"n", 2U}});
        const auto three = gr::trigger_test::scriptsAtChunkSizes<TakeN<float>>("a T:b c d e |", {{"filter", std::string("start")}, {"n", 3U}});
        expect(neq(two[0], three[0])) << "and the difference is visible in what comes out";
    };
};

int main() { /* tests are statically executed */ }
